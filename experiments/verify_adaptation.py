"""Audit the saved adaptation run independently of its sequential detector."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from harness import dayahead as da
from .adaptation_results import METHODS, MONTHLY_SCORE
from .backtest_adaptation import BUDGET, DAY, END, OUTPUT, PERIODIC_DAYS, START, TASKS


def verify_trace(trace, reference, thresholds, updates):
    calendar = pd.date_range(START, END, name="market_day")
    assert set(zip(trace.task, trace.strategy)) == {
        (task, strategy) for task in TASKS for strategy in ("frozen", "drift")
    }
    for (task, strategy), group in trace.groupby(["task", "strategy"]):
        group = group.set_index("market_day")
        assert group.index.equals(calendar)
        changes = updates.loc[updates.task.eq(task) & updates.strategy.eq(strategy)]
        version = np.searchsorted(changes.effective_day.to_numpy(), calendar.to_numpy(), side="right")
        np.testing.assert_array_equal(group.model_version_used, version)
        expected_signals = {}
        for reference_col, daily_col, stat, threshold_col, signal in (
            ("input", "input_distance", "input_rolling", "input_threshold", "input_alarm"),
            (task, "daily_error_pct", "error_rolling_pct", "error_threshold_pct", "error_alarm"),
        ):
            initial = reference[reference_col].tail(27).to_numpy()
            daily = group[daily_col].to_numpy()
            expected = np.full(len(daily), np.nan)
            sections = np.split(np.arange(len(daily)), np.flatnonzero(np.diff(version)) + 1)
            if reference_col == "input":
                sections = [np.arange(len(daily))]
            signal_days = np.zeros(len(daily), dtype=bool)
            seasonal_threshold = thresholds[reference_col].reindex(
                np.take(["summer", "autumn", "winter", "spring"], (group.index.month % 12) // 3)
            ).to_numpy()
            for part in sections:
                raw = daily[part]
                if part[0] == 0:
                    raw = np.r_[initial, raw]
                rolling = pd.Series(raw).rolling(28).mean().to_numpy()[-len(part):]
                expected[part] = rolling
                above = pd.Series(rolling > seasonal_threshold[part])
                full_run = above.rolling(7).sum().eq(7)
                signal_days[part] = full_run & ~full_run.shift(fill_value=False)
            np.testing.assert_allclose(expected, group[stat], atol=1e-10, rtol=0)
            np.testing.assert_allclose(seasonal_threshold, group[threshold_col], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(signal_days, group[signal])
            expected_signals[signal] = signal_days
        assert (group.action.eq("retrain") == group.index.isin(changes.decision_day)).all()
        alarm = expected_signals["input_alarm"] | expected_signals["error_alarm"]
        expected_reason = np.select(
            [expected_signals["input_alarm"] & expected_signals["error_alarm"],
             expected_signals["input_alarm"], expected_signals["error_alarm"]],
            ["input+error", "input", "error"], default="",
        )
        np.testing.assert_array_equal(group.alarm_sources.fillna("").to_numpy(), expected_reason)
        if strategy == "drift":
            triggered = group.loc[alarm & (version < BUDGET) & (group.index < END)]
            np.testing.assert_array_equal(triggered.index.to_numpy(), changes.decision_day.to_numpy())
            np.testing.assert_array_equal(triggered.alarm_sources.to_numpy(), changes.reason.to_numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output
    updates = pd.read_csv(out / "retraining.csv", parse_dates=[
        "decision_day", "effective_day", "train_first_day", "train_last_day",
    ])
    for (task, strategy), group in updates.groupby(["task", "strategy"]):
        assert len(group) <= BUDGET
        assert group.effective_day.is_monotonic_increasing and group.effective_day.is_unique
        assert group["update"].tolist() == list(range(1, len(group) + 1))
        assert group.effective_day.between(START, END).all()
        assert group.effective_day.eq(group.decision_day + DAY).all()
        assert group.train_last_day.eq(group.decision_day).all()
        assert group.train_first_day.eq(group.effective_day - pd.DateOffset(years=3)).all()
        assert group.n_halfhours.eq((group.effective_day - group.train_first_day).dt.days * 48).all()
        if strategy == "periodic":
            assert set(group.effective_day) == PERIODIC_DAYS

    assert set(updates.loc[updates.strategy.eq("periodic"), "task"]) == set(TASKS)
    trace = pd.read_csv(out / "detector_daily.csv", parse_dates=["market_day"])
    reference = pd.read_csv(out / "reference_daily.csv", index_col="market_day", parse_dates=True)
    thresholds = pd.read_csv(out / "thresholds.csv", index_col="season")
    assert reference.index.equals(pd.date_range("2014-01-01", "2014-12-31", name="market_day"))
    for i, label in enumerate(["summer", "autumn", "winter", "spring"]):
        windows = reference.rolling(28).mean().dropna()
        values = windows[(windows.index.month % 12) // 3 == i]
        assert len(values) == thresholds.at[label, "n_windows"]
        np.testing.assert_allclose(
            np.quantile(values.to_numpy(), .95, axis=0),
            thresholds.loc[label, reference.columns].to_numpy(),
        )
    verify_trace(trace, reference, thresholds, updates)

    monthly = pd.read_csv(out / "monthly_metrics.csv")
    yearly = pd.read_csv(out / "yearly_metrics.csv")
    score_functions = {
        "demand": {"MAPE": da.mape, "MAE": da.mae},
        "price": {"accuracy": da.accuracy, "balanced_accuracy": da.balanced_accuracy},
    }
    for task in TASKS:
        result = pd.read_csv(out / f"{task}_predictions.csv.gz", parse_dates=["timestamp", "market_day"])
        assert result.timestamp.equals(pd.Series(pd.date_range(
            START + pd.Timedelta(minutes=30), END + DAY, freq="30min"
        ), name="timestamp"))
        np.testing.assert_array_equal(result.market_day, da.market_day(pd.DatetimeIndex(result.timestamp)))
        assert np.isfinite(result[["actual", *METHODS]].to_numpy()).all()
        if task == "price":
            assert result[["actual", *METHODS]].isin([0, 1]).all().all()
        for strategy in ("periodic", "drift"):
            dates = updates.loc[updates.task.eq(task) & updates.strategy.eq(strategy), "effective_day"]
            first = dates.min() if len(dates) else END + DAY
            early = result.market_day.lt(first)
            np.testing.assert_array_equal(result.loc[early, strategy], result.loc[early, "frozen"])
        for frequency, table in (("M", monthly), ("Y", yearly)):
            provided = table.loc[table.task.eq(task)].copy()
            provided["period"] = provided.period.astype(str)
            provided = provided.set_index(["period", "method"])
            for col, calculate in score_functions[task].items():
                reconstructed = pd.Series({
                    (str(period), name): calculate(data.actual, data[name])
                    for period, data in result.groupby(result.market_day.dt.to_period(frequency))
                    for name in METHODS
                })
                assert provided.index.is_unique and len(provided) == len(reconstructed)
                np.testing.assert_allclose(provided[col], reconstructed.reindex(provided.index))
        monthly_task = monthly.loc[monthly.task.eq(task)]
        assert monthly_task.ci_metric.eq(MONTHLY_SCORE[task]).all()
        for lower, upper in (("iid_lower", "iid_upper"), ("day_bootstrap_lower", "day_bootstrap_upper")):
            assert np.isfinite(monthly_task[[lower, upper]].to_numpy()).all()
            assert monthly_task[lower].le(monthly_task[upper]).all()

    verified = {
        "status": "passed",
        "days_per_task": (END - START).days + 1,
        "half_hours_per_task": ((END - START).days + 1) * 48,
        "checks": [
            "All predictions use the official market-day calendar and contain valid task outputs",
            "All rolling statistics and alerts independently replayed with vectorized windows",
            "Thresholds rebuilt using only complete 2014 windows",
            "Alert actions, model versions, dates, three-year training windows and six-update limits verified",
            "All four monthly and annual metrics recomputed and both monthly intervals checked for valid bounds",
        ],
    }
    (out / "verification.json").write_text(json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verified, indent=2))


if __name__ == "__main__":
    main()
