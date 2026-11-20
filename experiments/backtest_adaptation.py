"""Backtest frozen, calendar-based and drift-triggered model updates.

Run from the kit root: python -m experiments.backtest_adaptation
"""
import argparse
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from harness import dayahead as da
from . import demand_models, drift_detectors, price_models
from .stage1_data import (
    DATA_PATH, ROOT, TARGET_COLUMNS, VALIDATION_START, VALIDATION_END,
    baseline_predictions, build_features,
)

START = pd.Timestamp("2015-01-01")
END = pd.Timestamp("2026-10-31")
DAY = pd.Timedelta(days=1)
BUDGET = 6
TRAIN_YEARS = 3
PERIODIC_DAYS = {pd.Timestamp(year, 1, 1) for year in range(2016, 2027, 2)}
TASKS = {"demand": "mlp", "price": "hist_gbc"}
STRATEGIES = ("frozen", "periodic", "drift")
BASELINES = ("yesterday", "last_week")
OUTPUT = ROOT / "outputs" / "adaptation"
FROZEN = ROOT / "outputs" / "frozen_backtest"


def load_examples(path):
    data = da.load_dataset(path)
    start = pd.Timestamp("2010-01-01")
    expected = pd.date_range(start + pd.Timedelta(minutes=30), END + DAY, freq="30min")
    if not data.index.equals(expected) or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Expected one complete course dataset at 30-minute frequency.")

    # Shared features only look up prices and demand on earlier market days.
    # Precomputing these deterministic features involves no learned transforms.
    market_days = da.market_day(data.index)
    features = build_features(data, data[da.WEATHER_COLS]).loc[market_days >= start + 7 * DAY]
    truth = pd.DataFrame({
        TARGET_COLUMNS["demand"]: data.loc[features.index, "nsw_demand"],
        TARGET_COLUMNS["price"]: (
            data.loc[features.index, "nsw_price"] > features["previous_price_mean"]
        ).astype(int),
    })
    check_days = [VALIDATION_START, START, *sorted(PERIODIC_DAYS), END]
    for day in check_days:
        stamp = da.market_day(features.index) == day
        history, weather = da.make_inputs(data, day)
        pd.testing.assert_frame_equal(features.loc[stamp], build_features(history.tail(7 * 48), weather))
        pd.testing.assert_frame_equal(truth.loc[stamp], da.targets(data, day))
    return features, truth, check_days


def daily_error(task, truth, predicted):
    if task == "demand":
        return da.mape(truth, predicted)
    return 100.0 * float(np.mean(truth != predicted))


def fit_update(task, strategy, features, truth, day, version, causes):
    factory = demand_models if task == "demand" else price_models
    model = factory.build_models()[TASKS[task]]
    first = day - pd.DateOffset(years=TRAIN_YEARS)
    start, stop = features.index.searchsorted(
        [first + pd.Timedelta(minutes=30), day + pd.Timedelta(minutes=30)]
    )
    train_x = features.iloc[start:stop]
    train_y = truth.iloc[start:stop]
    if train_x.index[0] != first + pd.Timedelta(minutes=30) or train_x.index[-1] != day:
        raise ValueError(f"Incomplete training window before {day.date()}")
    print(
        f"{task} {strategy} update {version}: {first.date()} to {(day - DAY).date()}, "
        f"effective {day.date()} ({causes})", flush=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_x, train_y)
    event = {
        "task": task, "strategy": strategy, "model": TASKS[task], "update": version,
        "decision_day": (day - DAY).date(), "effective_day": day.date(), "reason": causes,
        "train_first_day": first.date(), "train_last_day": (day - DAY).date(),
        "n_halfhours": len(train_x), "fit_warnings": " | ".join(str(w.message) for w in caught),
    }
    return model, event


def simulate(task, strategy, initial, features, truth, days, distance, reference, thresholds, frozen):
    start = features.index.searchsorted(START + pd.Timedelta(minutes=30))
    errors = drift_detectors.Detector(thresholds[task], reference[task].tail(27))
    inputs = drift_detectors.Detector(thresholds["input"], reference["input"].tail(27))
    model = initial
    predictions = np.empty((len(days), 48), dtype=float if task == "demand" else int)
    updates, trace = [], []
    for i, day in enumerate(days):
        if strategy == "periodic" and day in PERIODIC_DAYS:
            model, event = fit_update(task, strategy, features, truth, day, len(updates) + 1, "calendar")
            updates.append(event)
        rows = slice(start + 48 * i, start + 48 * (i + 1))
        deployed_version = len(updates)
        if deployed_version == 0:
            pred = frozen[i]
        else:
            pred = model.predict(features.iloc[rows])
        if pred.shape != (48,) or not np.isfinite(pred).all():
            raise ValueError(f"Invalid output for {task} {strategy} on {day.date()}")
        if task == "price" and not np.isin(pred, [0, 1]).all():
            raise ValueError(f"Invalid classes on {day.date()}")
        predictions[i] = pred
        observed_error = daily_error(task, truth.iloc[rows].to_numpy(), pred)
        if strategy == "periodic":
            continue

        input_stat, input_limit, input_alarm = inputs.observe(day, distance.at[day])
        error_stat, error_limit, error_alarm = errors.observe(day, observed_error)
        alarm_sources = "+".join(
            name for name, alarm in (("input", input_alarm), ("error", error_alarm)) if alarm
        )
        action = ""
        if alarm_sources:
            if strategy == "frozen":
                action = "monitor_only"
            elif len(updates) == BUDGET:
                action = "budget_exhausted"
            elif day == days[-1]:
                action = "outside_simulation"
            else:
                model, event = fit_update(
                    task, strategy, features, truth, day + DAY, len(updates) + 1, alarm_sources
                )
                updates.append(event)
                errors.reset()
                action = "retrain"
        trace.append({
            "task": task, "strategy": strategy, "market_day": day.date(),
            "model_version_used": deployed_version, "input_distance": distance.at[day],
            "daily_error_pct": observed_error,
            "input_rolling": input_stat, "input_threshold": input_limit, "input_alarm": input_alarm,
            "error_rolling_pct": error_stat, "error_threshold_pct": error_limit, "error_alarm": error_alarm,
            "alarm_sources": alarm_sources, "action": action,
        })
    return predictions.ravel(), updates, trace


def verify_frozen(frame, task):
    """Check all labels and baselines, and record prediction parity across runtimes."""
    path = FROZEN / f"{task}_predictions.csv.gz"
    saved = pd.read_csv(path, parse_dates=["timestamp", "market_day"])
    for column in ("timestamp", "market_day", "actual", *BASELINES):
        np.testing.assert_array_equal(frame[column], saved[column])
    predicted = frame["frozen"].to_numpy()
    previous = saved[TASKS[task]].to_numpy()
    if task == "demand":
        np.testing.assert_allclose(predicted, previous, rtol=1e-10, atol=1e-8)
        return {"max_abs_difference_mw": float(np.max(np.abs(predicted - previous)))}
    changed = predicted != previous
    return {"different_class_predictions": int(changed.sum()),
            "affected_timestamps": frame.loc[changed, "timestamp"].astype(str).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--day-types", type=Path, default=ROOT / "day_types.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    features, truth, check_days = load_examples(args.data)
    days = pd.date_range(START, END)
    dates = da.market_day(features.index)
    ref = (dates >= VALIDATION_START) & (dates < VALIDATION_END)
    test = dates >= START
    types = pd.read_csv(args.day_types, parse_dates=["market_day"]).set_index("market_day")
    distance = types["nearest_center_distance"]
    detector_dates = pd.date_range(VALIDATION_START, END)
    if not distance.index.is_unique or not np.isfinite(distance.reindex(detector_dates)).all():
        raise ValueError("Expected one finite day-type distance per calibration and simulation day.")
    reference = distance.reindex(pd.date_range(VALIDATION_START, START - DAY)).to_frame("input")

    all_updates, all_trace = [], []
    frozen_parity = {}
    initial_models = {
        "demand": joblib.load(ROOT / "artifacts" / "demand_mlp_frozen.joblib"),
        "price": joblib.load(FROZEN / "price_hist_gbc_frozen.joblib"),
    }
    initial_hashes = {task: joblib.hash(model) for task, model in initial_models.items()}
    with threadpool_limits(limits=2):
        for task, model in initial_models.items():
            y = truth[TARGET_COLUMNS[task]]
            y_ref = y.loc[ref].to_numpy().reshape(-1, 48)
            p_ref = model.predict(features.loc[ref]).reshape(-1, 48)
            reference[task] = [
                daily_error(task, actual, predicted) for actual, predicted in zip(y_ref, p_ref)
            ]
        thresholds, counts = drift_detectors.calibrate(reference)
        thresholds.assign(n_windows=counts).rename_axis("season").to_csv(args.output / "thresholds.csv")
        reference.rename_axis("market_day").to_csv(args.output / "reference_daily.csv")
        for task, model in initial_models.items():
            y = truth[TARGET_COLUMNS[task]]
            test_x = features.loc[test]
            frame = pd.DataFrame({
                "timestamp": test_x.index, "market_day": dates[test], "actual": y.loc[test].to_numpy(),
                **{name: values.to_numpy() for name, values in baseline_predictions(test_x, task).items()},
            })
            # The frozen estimator has no state between predict calls. Batching preserves
            # the causal feature rows. Compare with the prior run, which may differ numerically.
            frozen = model.predict(test_x).reshape(-1, 48)
            frame["frozen"] = frozen.ravel()
            frozen_parity[task] = verify_frozen(frame, task)
            print(f"Archived frozen comparison for {task}: {frozen_parity[task]}", flush=True)
            for strategy in STRATEGIES:
                print(f"Simulating {task} / {strategy}", flush=True)
                predictions, updates, trace = simulate(
                    task, strategy, model, features, y, days, distance, reference, thresholds, frozen
                )
                frame[strategy] = predictions
                all_updates.extend(updates)
                all_trace.extend(trace)
            frame.to_csv(
                args.output / f"{task}_predictions.csv.gz", index=False,
                compression={"method": "gzip", "mtime": 0},
            )
    if initial_hashes != {task: joblib.hash(model) for task, model in initial_models.items()}:
        raise RuntimeError("The frozen reference estimators were altered.")

    pd.DataFrame(all_updates).to_csv(args.output / "retraining.csv", index=False)
    trace = pd.DataFrame(all_trace)
    trace.to_csv(args.output / "detector_daily.csv", index=False)
    trace.loc[trace.alarm_sources.ne("")].to_csv(args.output / "alarms.csv", index=False)
    manifest = {
        "simulation": [str(START.date()), str(END.date())],
        "reference_year": 2014, "models": TASKS, "days": len(days),
        "training_years": TRAIN_YEARS, "budget_per_model_and_strategy": BUDGET,
        "rolling_days": drift_detectors.WINDOW, "persistence_days": drift_detectors.PERSISTENCE,
        "reference_quantile": drift_detectors.QUANTILE,
        "official_boundary_checks": [str(day.date()) for day in check_days],
        "archived_frozen_comparison": frozen_parity,
        "platform": platform.system(),
        "versions": {
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
        },
    }
    (args.output / "simulation.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Simulation completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
