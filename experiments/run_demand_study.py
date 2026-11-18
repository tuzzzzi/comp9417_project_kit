"""Reproduce A's chronological model selection, diagnostics and frozen artifacts.

Only 2010-2014 market days enter this study. The full released CSV is identified
by a SHA256, but 2015+ outcomes do not enter fitting, selection or diagnostics.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from harness import dayahead as da
from .demand_models import CONFIG_PATH, DEFAULT_MLP, make_mlp, make_ridge
from .stage1_data import ROOT, VALIDATION_START, baseline_predictions, load_stage1

DEFAULT_DATA = ROOT.parent / "comp9417_project_kit/data/nem_weather_halfhourly_2010_2026.csv"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual, predicted):
    return {"MAPE": da.mape(actual, predicted), "MAE": da.mae(actual, predicted)}


def fit_one(identifier, family, settings, train_x, train_y, valid_x, valid_y, output):
    estimator = make_ridge(**settings) if family == "ridge" else make_mlp(**settings)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = np.asarray(estimator.predict(valid_x), dtype=float)
    predict_seconds = time.perf_counter() - started
    if prediction.shape != (len(valid_x),) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid predictions from {identifier}")
    train_score = metrics(train_y, estimator.predict(train_x))
    valid_score = metrics(valid_y, prediction)
    row = {
        "candidate": identifier, "family": family,
        "parameters": json.dumps(settings, sort_keys=True),
        "train_MAPE": train_score["MAPE"], "train_MAE": train_score["MAE"],
        **valid_score, "fit_seconds": fit_seconds,
        "validation_predict_seconds": predict_seconds,
        "convergence_warning": any(isinstance(w.message, ConvergenceWarning) for w in caught),
        "warning_messages": " | ".join(str(w.message) for w in caught),
    }
    if family == "mlp":
        network = estimator.regressor_.named_steps["mlpregressor"]
        row["epochs"] = network.n_iter_
        row["final_scaled_training_loss"] = network.loss_
        pd.DataFrame({"epoch": np.arange(1, len(network.loss_curve_) + 1),
                      "scaled_training_loss": network.loss_curve_}).to_csv(
            output / "loss_curves" / f"{identifier}.csv", index=False)
    print(f"{identifier}: MAPE={row['MAPE']:.6f}% MAE={row['MAE']:.3f} MW "
          f"fit={fit_seconds:.2f}s warnings={row['convergence_warning']}", flush=True)
    return estimator, prediction, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage1")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "loss_curves").mkdir(exist_ok=True)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)

    features, truth = load_stage1(args.data)
    days = da.market_day(features.index)
    training = days < VALIDATION_START
    train_x, valid_x = features.loc[training], features.loc[~training]
    labels = truth.nsw_demand_mw
    train_y, valid_y = labels.loc[training], labels.loc[~training]
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Features must be complete and finite")
    if len(train_x) != 69792 or len(valid_x) != 17520:
        raise ValueError("Unexpected course data coverage")
    print(f"Training {train_x.shape}; validation {valid_x.shape}; seed=42", flush=True)

    candidates = [(f"ridge_alpha_{alpha:g}", "ridge", {"alpha": alpha})
                  for alpha in (1.0, 10.0, 100.0)]
    for name, layers, alpha in [
        ("mlp_32", [32], 0.0001),
        ("mlp_64", [64], 0.0001),
        ("mlp_64_32", [64, 32], 0.0001),
        ("mlp_64_32_reg", [64, 32], 0.001),
    ]:
        candidates.append((name, "mlp", {**DEFAULT_MLP,
                                           "hidden_layer_sizes": layers, "alpha": alpha}))
    # The first 300-epoch pilot reached the iteration cap. This additional
    # candidate checks optimization sensitivity without looking beyond 2014.
    candidates.append(("mlp_64_long", "mlp", {**DEFAULT_MLP,
                       "hidden_layer_sizes": [64], "max_iter": 1200}))

    fitted, forecasts, records = {}, {}, []
    with threadpool_limits(limits=2):
        for identifier, family, settings in candidates:
            model, prediction, row = fit_one(identifier, family, settings,
                                             train_x, train_y, valid_x, valid_y, output)
            fitted[identifier], forecasts[identifier] = model, prediction
            records.append(row)
            pd.DataFrame(records).to_csv(output / "demand_tuning.csv", index=False)

        tuning = pd.DataFrame(records)
        winners = {family: tuning.loc[tuning.family == family].sort_values(
            ["MAPE", "MAE", "candidate"]).iloc[0] for family in ("ridge", "mlp")}
        selected = {family: fitted[row.candidate] for family, row in winners.items()}
        chosen_config = {
            "schema_version": 1,
            "training_market_days": ["2010-01-08", "2013-12-31"],
            "validation_market_days": ["2014-01-01", "2014-12-31"],
            "selection_metric": "2014 MAPE; tie-break MAE then candidate name",
            "ridge_alpha": json.loads(winners["ridge"].parameters)["alpha"],
            "mlp_params": json.loads(winners["mlp"].parameters),
            "selected_candidates": {k: str(v.candidate) for k, v in winners.items()},
            "feature_names": list(features.columns),
        }
        write_json(CONFIG_PATH, chosen_config)
        write_json(output / "demand_model_config.json", chosen_config)
        for name, estimator in selected.items():
            joblib.dump(estimator, artifacts / f"demand_{name}_frozen.joblib", compress=3)

        predictions = {k: np.asarray(v) for k, v in baseline_predictions(valid_x, "demand").items()}
        predictions.update({family: forecasts[row.candidate] for family, row in winners.items()})
        scores = pd.DataFrame([{"model": name, **metrics(valid_y, p)}
                               for name, p in predictions.items()])
        baseline_mape = float(scores.loc[scores.model == "yesterday", "MAPE"].iloc[0])
        scores["MAPE_improvement_vs_yesterday_pct"] = 100 * (1 - scores.MAPE / baseline_mape)
        scores.to_csv(output / "demand_validation.csv", index=False)
        table = pd.DataFrame({"market_day": da.market_day(valid_x.index),
                              "actual": valid_y, **predictions}, index=valid_x.index)
        table.to_csv(output / "demand_predictions.csv", index_label="timestamp")
        long = table.reset_index(names="timestamp").melt(
            id_vars=["timestamp", "market_day", "actual"], value_vars=list(predictions),
            var_name="model", value_name="pred_demand").rename(columns={"actual": "y_demand"})
        long.to_csv(output / "demand_predictions_long.csv", index=False)

        seed_rows, seed_predictions = [], {}
        primary = winners["mlp"].to_dict()
        seed_rows.append({"seed": 42, **{k: primary[k] for k in
                          ["MAPE", "MAE", "fit_seconds", "epochs", "convergence_warning"]}})
        seed_predictions["seed_42"] = predictions["mlp"]
        for seed in (0, 2026):
            settings = {**chosen_config["mlp_params"], "random_state": seed}
            _, p, row = fit_one(f"selected_mlp_seed_{seed}", "mlp", settings,
                                 train_x, train_y, valid_x, valid_y, output)
            seed_rows.append({"seed": seed, **{k: row[k] for k in
                              ["MAPE", "MAE", "fit_seconds", "epochs", "convergence_warning"]}})
            seed_predictions[f"seed_{seed}"] = p
        pd.DataFrame(seed_rows).to_csv(output / "demand_seed_stability.csv", index=False)
        pd.DataFrame({"actual": valid_y, **seed_predictions}).to_csv(
            output / "demand_seed_predictions.csv", index_label="timestamp")

        groups = {
            "weather": list(da.WEATHER_COLS),
            "calendar": ["slot_sin", "slot_cos", "weekend", "season_sin", "season_cos"],
            "demand_lags": ["demand_lag1", "demand_lag7"],
            "price_history": ["price_lag1", "price_lag7", "previous_price_mean"],
        }
        ablation_rows = [{"model": family, "removed_group": "none", "n_features": len(features.columns),
                          **metrics(valid_y, predictions[family])} for family in selected]
        ablation_predictions = {}
        for group, removed in groups.items():
            columns = [c for c in features.columns if c not in removed]
            for family in ("ridge", "mlp"):
                settings = ({"alpha": chosen_config["ridge_alpha"]} if family == "ridge"
                            else chosen_config["mlp_params"])
                _, p, row = fit_one(f"ablation_{family}_without_{group}", family, settings,
                                     train_x[columns], train_y, valid_x[columns], valid_y, output)
                ablation_rows.append({"model": family, "removed_group": group,
                                      "n_features": len(columns), "MAPE": row["MAPE"],
                                      "MAE": row["MAE"], "fit_seconds": row["fit_seconds"],
                                      "convergence_warning": row["convergence_warning"]})
                ablation_predictions[f"{family}_without_{group}"] = p
                pd.DataFrame(ablation_rows).to_csv(output / "demand_feature_ablation.csv", index=False)
        pd.DataFrame({"actual": valid_y, **ablation_predictions}).to_csv(
            output / "demand_ablation_predictions.csv", index_label="timestamp")

    metric_records = scores.loc[scores.model.isin(selected)].sort_values(["MAPE", "MAE", "model"])
    chosen_config["recommended_model"] = str(metric_records.iloc[0].model)
    write_json(output / "demand_model_config.json", chosen_config)
    write_json(CONFIG_PATH, chosen_config)
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "threadpoolctl"]}
    source_paths = [Path("experiments") / name for name in
                    ["demand_models.py", "demand_config.json", "stage1_data.py",
                     "validate_models.py", "run_demand_study.py"]]
    manifest = {
        "python": sys.version, "platform": platform.platform(), "versions": versions,
        "data_filename": args.data.name, "data_sha256": sha256(args.data),
        "data_usage": "Only market days before 2015-01-01 enter fitting, selection and diagnostics",
        "train_rows": len(train_x), "validation_rows": len(valid_x),
        "feature_names": list(features.columns),
        "train_first_timestamp": str(train_x.index.min()), "train_last_timestamp": str(train_x.index.max()),
        "validation_first_timestamp": str(valid_x.index.min()), "validation_last_timestamp": str(valid_x.index.max()),
        "thread_limit": 2, "selection_seed": 42, "stability_seeds": [42, 0, 2026],
        "selection_candidates": len(candidates), "refit_on_2014": False,
        "ablation_role": "diagnostic only; shared 22-feature representation retained",
        "seed_role": "diagnostic only; seed 42 retained rather than selecting the best seed",
        "recommended_model": chosen_config["recommended_model"],
        "source_sha256": {str(p): sha256(ROOT / p) for p in source_paths},
        "artifact_sha256": {p.name: sha256(p) for p in sorted(artifacts.glob("*.joblib"))},
    }
    write_json(output / "demand_run_manifest.json", manifest)
    print("\nFinal 2014 validation:\n" + scores.to_string(index=False), flush=True)
    print(f"Recommended model: {chosen_config['recommended_model']}", flush=True)


if __name__ == "__main__":
    main()
