"""Check A's saved outputs against source records, daily inputs and loaded models."""

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from harness import dayahead as da
from . import demand_models
from .demand_models import build_models
from .run_demand_study import DEFAULT_DATA
from .stage1_data import ROOT, VALIDATION_START, build_features, load_stage1


def independent_scores(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    return np.array([100 * np.mean(np.abs(y - prediction) / np.abs(y)),
                     np.mean(np.abs(y - prediction))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage1")
    parser.add_argument("--reproduced", type=Path,
                        help="Optional output folder from unchanged experiments.validate_models")
    args = parser.parse_args()
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)
        print("PASS", name, flush=True)

    check("Python major/minor is 3.11", sys.version_info[:2] == (3, 11))
    expected = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_-]+)==([^\s]+)", line.strip())
        if match:
            expected[match[1]] = match[2]
    check("Every course-pinned dependency version matches",
          all(importlib.metadata.version(k) == v for k, v in expected.items()))

    data = da.load_dataset(args.data)
    expected_index = pd.date_range("2010-01-01 00:30", "2026-11-01 00:00", freq="30min")
    check("Released CSV has exactly the expected complete timestamp grid", data.index.equals(expected_index))
    check("Released CSV has no missing values", not data.isna().any().any())
    features, truth = load_stage1(args.data)
    train = da.market_day(features.index) < VALIDATION_START
    valid_x = features.loc[~train]
    check("Training and validation have 69,792 and 17,520 rows", (int(train.sum()), int((~train).sum())) == (69792, 17520))
    check("Only 22 finite features are used", features.shape[1] == 22 and np.isfinite(features.to_numpy()).all())
    check("2015+ market days excluded from feature table", da.market_day(features.index).max() == pd.Timestamp("2014-12-31"))

    predictions = pd.read_csv(args.output / "demand_predictions.csv", parse_dates=["timestamp", "market_day"]).set_index("timestamp")
    scores = pd.read_csv(args.output / "demand_validation.csv").set_index("model")
    check("Saved prediction index equals all 2014 validation timestamps", predictions.index.equals(valid_x.index))
    check("Every prediction day has exactly 48 intervals", predictions.groupby("market_day").size().eq(48).all())
    check("Saved market days use interval-start dates", np.array_equal(predictions.market_day.to_numpy(), da.market_day(predictions.index).to_numpy()))
    check("Ground truth equals released NSW demand", np.array_equal(predictions.actual.to_numpy(), data.nsw_demand.reindex(predictions.index).to_numpy()))
    check("Validation includes midnight of 2015-01-01 as 2014-12-31", predictions.index[-1] == pd.Timestamp("2015-01-01") and predictions.market_day.iloc[-1] == pd.Timestamp("2014-12-31"))
    for model in ["yesterday", "last_week", "ridge", "mlp"]:
        p = predictions[model].to_numpy()
        check(f"{model} predictions finite and within submission demand range", np.isfinite(p).all() and ((p >= 0) & (p <= 20000)).all())
        calculated = independent_scores(predictions.actual, p)
        check(f"{model} MAPE and MAE independently recomputed", np.allclose(calculated, scores.loc[model, ["MAPE", "MAE"]].to_numpy(float), rtol=0, atol=1e-8))
        if model in ["yesterday", "last_week"]:
            lag = 1 if model == "yesterday" else 7
            expected_values = data.nsw_demand.reindex(predictions.index - pd.Timedelta(days=lag)).to_numpy()
            check(f"{model} baseline equals raw same-slot lag", np.array_equal(p, expected_values))

    models = {name: joblib.load(ROOT / "artifacts" / f"demand_{name}_frozen.joblib") for name in ["ridge", "mlp"]}
    timings = []
    with threadpool_limits(limits=2):
        for name, model in models.items():
            check(f"{name} saved model reproduces all saved predictions", np.allclose(model.predict(valid_x), predictions[name], rtol=0, atol=1e-8))
            input_scaler = model.named_steps["standardscaler"] if name == "ridge" else model.regressor_.named_steps["standardscaler"]
            check(f"{name} input scaler fitted on training rows only", int(input_scaler.n_samples_seen_) == 69792 and np.allclose(input_scaler.mean_, features.loc[train].mean().to_numpy()))
            if name == "mlp":
                check("MLP target scaler fitted on training targets only", int(model.transformer_.n_samples_seen_) == 69792 and np.isclose(model.transformer_.mean_[0], truth.nsw_demand_mw.loc[train].mean()))
                network = model.regressor_.named_steps["mlpregressor"]
                check("MLP retains seed 42 and disables automatic validation split", network.random_state == 42 and network.early_stopping is False)

        # Boundary, winter and year-end days are selected by date, not model errors.
        # Compare offline construction with the actual history/weather-only interface.
        for day in ["2014-01-01", "2014-07-01", "2014-12-31"]:
            history, weather = da.make_inputs(data, day)
            daily_x = build_features(history, weather)
            check(f"Daily inputs match offline inputs on {day}", np.allclose(daily_x, valid_x.loc[weather.index], rtol=0, atol=1e-9))
            poisoned = data.copy()
            future = da.market_day(poisoned.index) >= pd.Timestamp(day)
            poisoned.loc[future, da.MARKET_COLS] = 1e30
            poison_x = build_features(poisoned, weather)
            check(f"Future market poisoning leaves inputs unchanged on {day}", np.allclose(poison_x, daily_x, rtol=0, atol=1e-9))
            for name, model in models.items():
                start = time.perf_counter()
                p = model.predict(daily_x)
                seconds = time.perf_counter() - start
                timings.append({"model": name, "market_day": day, "predict_48_seconds": seconds})
                check(f"{name} day prediction matches saved batch result on {day}", np.allclose(p, predictions.loc[weather.index, name], rtol=0, atol=1e-8))

    first, second = build_models(), build_models()
    check("Model factory returns independent fresh estimators", all(first[n] is not second[n] for n in first))
    check("Factory Ridge hyperparameters equal the saved model", first["ridge"].named_steps["ridge"].get_params() == models["ridge"].named_steps["ridge"].get_params())
    check("Factory MLP hyperparameters equal the saved model", first["mlp"].regressor.named_steps["mlpregressor"].get_params() == models["mlp"].regressor_.named_steps["mlpregressor"].get_params())
    with tempfile.TemporaryDirectory(prefix="a-config-check-") as temporary:
        config_path = Path(temporary) / "demand_config.json"
        with patch.object(demand_models, "CONFIG_PATH", config_path):
            missing_rejected = False
            try:
                build_models()
            except FileNotFoundError:
                missing_rejected = True
            check("Missing configuration fails explicitly instead of changing the model", missing_rejected)
            config_path.write_text(json.dumps({"schema_version": 1, "ridge_alpha": 10, "mlp_params": {}}))
            incomplete_rejected = False
            try:
                build_models()
            except ValueError:
                incomplete_rejected = True
            check("Incomplete configuration fails explicitly", incomplete_rejected)
    long = pd.read_csv(args.output / "demand_predictions_long.csv")
    check("C handoff has four complete model series without duplicates", len(long) == 4 * 17520 and not long.duplicated(["timestamp", "model"]).any())
    check("C handoff truth and prediction fields are finite", np.isfinite(long[["y_demand", "pred_demand"]].to_numpy()).all())

    seeds = pd.read_csv(args.output / "demand_seed_stability.csv").set_index("seed")
    seed_predictions = pd.read_csv(args.output / "demand_seed_predictions.csv")
    for seed in [42, 0, 2026]:
        check(f"Seed {seed} scores recomputed", np.allclose(independent_scores(seed_predictions.actual, seed_predictions[f"seed_{seed}"]), seeds.loc[seed, ["MAPE", "MAE"]].to_numpy(float), atol=1e-8))
    ablation = pd.read_csv(args.output / "demand_feature_ablation.csv")
    ablation_predictions = pd.read_csv(args.output / "demand_ablation_predictions.csv")
    for row in ablation.itertuples():
        if row.removed_group == "none":
            continue
        key = f"{row.model}_without_{row.removed_group}"
        check(f"Ablation {key} scores recomputed", np.allclose(independent_scores(ablation_predictions.actual, ablation_predictions[key]), [row.MAPE, row.MAE], atol=1e-8))

    monthly = pd.read_csv(args.output / "demand_monthly.csv")
    check("All four methods have 12 monthly estimates", len(monthly) == 48 and monthly.groupby("model").size().eq(12).all())
    monthly_matches = True
    iid_matches = True
    for row in monthly.itertuples():
        mask = predictions.market_day.dt.strftime("%Y-%m") == row.month
        y = predictions.actual.loc[mask].to_numpy()
        p = predictions[row.model].loc[mask].to_numpy()
        result = independent_scores(y, p)
        monthly_matches &= bool(np.allclose(result, [row.MAPE, row.MAE], atol=1e-8))
        ape = 100 * np.abs(y - p) / y
        width = 1.96 * np.std(ape, ddof=1) / np.sqrt(len(ape))
        iid_matches &= bool(np.allclose([result[0] - width, result[0] + width],
                                        [row.iid_lower, row.iid_upper], atol=1e-8))
    check("Monthly chart means independently recomputed", monthly_matches)
    check("Monthly IID intervals independently recomputed", iid_matches)
    check("Daily bootstrap bounds finite, ordered and based on 2000 replicates",
          np.isfinite(monthly[["day_bootstrap_lower", "day_bootstrap_upper"]].to_numpy()).all()
          and (monthly.day_bootstrap_lower <= monthly.day_bootstrap_upper).all()
          and monthly.bootstrap_replicates.eq(2000).all())

    if args.reproduced:
        repeat = pd.read_csv(args.reproduced / "demand_predictions.csv")
        check("Unchanged group validator retraining reproduces every prediction", np.allclose(repeat[["actual", "yesterday", "last_week", "ridge", "mlp"]], predictions.reset_index()[["actual", "yesterday", "last_week", "ridge", "mlp"]], rtol=0, atol=1e-8))
    manifest = json.loads((args.output / "demand_run_manifest.json").read_text())
    check("Source CSV SHA256 matches the training run", hashlib.sha256(args.data.read_bytes()).hexdigest() == manifest["data_sha256"])
    for filename, digest in manifest["source_sha256"].items():
        post_change = manifest.get("post_training_source_changes", {}).get(filename)
        if post_change:
            check(f"Post-training change preserves original provenance for {filename}", post_change["prior_sha256"] == digest)
            digest = post_change["sha256"]
        check(f"Recorded source hash matches {filename}", hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest)
    for filename, digest in manifest["artifact_sha256"].items():
        check(f"Recorded model hash matches {filename}", hashlib.sha256((ROOT / "artifacts" / filename).read_bytes()).hexdigest() == digest)
    pd.DataFrame(timings).to_csv(args.output / "demand_daily_prediction_timing.csv", index=False)
    (args.output / "demand_checks.json").write_text(json.dumps({
        "status": "passed", "checks": checks,
        "scope": "2014 validation and three sampled daily inference/leakage boundaries; not final live-week sandbox",
        "retrained_with_group_validator": bool(args.reproduced),
    }, indent=2) + "\n")
    print(f"Completed {len(checks)} checks.")


if __name__ == "__main__":
    main()
