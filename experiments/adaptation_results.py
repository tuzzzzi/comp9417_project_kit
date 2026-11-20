"""Generate metrics and figures from the saved Stage 2 predictions and logs."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest_adaptation import BASELINES, DAY, END, OUTPUT, START, STRATEGIES, TASKS
from .backtest_frozen import intervals, metric, plt

METHODS = (*STRATEGIES, *BASELINES)
PRIMARY = {"demand": "MAPE", "price": "balanced_accuracy"}
MONTHLY_SCORE = {"demand": "MAPE", "price": "accuracy"}
SPLIT = pd.Timestamp("2025-01-01")
PERIODS = {"selection": (START, SPLIT - DAY), "holdout": (SPLIT, END), "all": (START, END)}
COLORS = {
    "frozen": "#55657a", "periodic": "#007b73", "drift": "#ad4900",
    "yesterday": "#3d395e", "last_week": "#777777",
}
REPLICATES = 2000


def score_periods(task, frame, dates, updates, rng):
    monthly, yearly, pooled = [], [], []
    for freq, rows in (("M", monthly), ("Y", yearly)):
        for period, group in frame.groupby(dates.dt.to_period(freq)):
            y = group.actual.to_numpy()
            for name in METHODS:
                pred = group[name].to_numpy()
                result = {
                    "task": task, "method": name, "period": str(period),
                    "n_days": len(y) // 48, **metric(task, y, pred),
                }
                if freq == "M":
                    result.update(intervals(y, pred, task, rng, 1000))
                rows.append(result)
    for label, (first, last) in PERIODS.items():
        group = frame.loc[dates.between(first, last)]
        y = group.actual.to_numpy()
        for name in METHODS:
            effective = updates.loc[updates.strategy.eq(name), "effective_day"]
            pooled.append({
                "task": task, "method": name, "period": label, "n_days": len(y) // 48,
                "updates_in_period": int(effective.between(first, last).sum()),
                "updates_as_of_end": int(effective.le(last).sum()),
                **metric(task, y, group[name].to_numpy()),
            })
    return monthly, yearly, pooled


def paired_effects(task, frame, rng):
    """Resample the same contiguous market days for both methods in a comparison."""
    comparisons = (*combinations(STRATEGIES, 2), ("yesterday", "periodic"), ("yesterday", "drift"))
    output = []
    for label, (first, last) in PERIODS.items():
        group = frame.loc[frame.market_day.between(first, last)]
        actual = group.actual.to_numpy().reshape(-1, 48)
        n_days = len(actual)
        # Circular moving blocks preserve serial dependence within sampled 14-day runs.
        starts = rng.integers(0, n_days, size=(REPLICATES, (n_days + 13) // 14))
        draws = ((starts[:, :, None] + np.arange(14)) % n_days).reshape(REPLICATES, -1)[:, :n_days]
        sampled = {}
        for name in METHODS:
            pred = group[name].to_numpy().reshape(-1, 48)
            if task == "demand":
                daily = (100 * np.abs(actual - pred) / actual).mean(axis=1)
                sampled[name] = daily[draws].mean(axis=1)
            else:
                recalls = []
                for c in (0, 1):
                    count = (actual == c).sum(axis=1)[draws].sum(axis=1)
                    correct = ((actual == c) & (pred == c)).sum(axis=1)[draws].sum(axis=1)
                    recalls.append(np.divide(correct, count, out=np.full(REPLICATES, np.nan), where=count > 0))
                sampled[name] = 100 * np.nanmean(recalls, axis=0)
        score = PRIMARY[task]
        for control, candidate in comparisons:
            delta = sampled[candidate] - sampled[control]
            observed = metric(task, group.actual, group[candidate])[score] - metric(task, group.actual, group[control])[score]
            if task == "demand":
                delta, observed = -delta, -observed
            low, high = np.percentile(delta, [2.5, 97.5])
            output.append({"task": task, "period": label, "candidate": candidate, "control": control,
                           "gain_pp": observed, "low_95_pp": low, "high_95_pp": high})
    return output


def select_strategy(pooled):
    rows = []
    for task, score in PRIMARY.items():
        scores = pooled.loc[pooled.task.eq(task) & pooled.method.isin(STRATEGIES)]
        ranked = scores.loc[scores.period.eq("selection")].sort_values(
            [score, "updates_as_of_end", "method"], ascending=[task == "demand", True, True]
        )
        chosen = ranked.iloc[0]
        holdout = scores.loc[scores.period.eq("holdout") & scores.method.eq(chosen.method)].iloc[0]
        rows.append({"task": task, "model": TASKS[task], "strategy": chosen.method, "metric": score,
                     "selection_score": chosen[score], "holdout_score": holdout[score],
                     "updates_by_2024": chosen.updates_as_of_end,
                     "updates_by_2026_oct": holdout.updates_as_of_end})
    return pd.DataFrame(rows)


def detector_evidence(trace):
    rows, alarm_windows = [], []
    for (task, strategy), group in trace.groupby(["task", "strategy"]):
        group = group.set_index("market_day").sort_index()
        for label, (first, last) in PERIODS.items():
            subset = group.loc[first:last]
            in_above = subset.input_rolling.gt(subset.input_threshold)
            ready = subset.error_rolling_pct.notna()
            err_above = subset.error_rolling_pct.gt(subset.error_threshold_pct)
            rows.append({
                "task": task, "strategy": strategy, "period": label, "days": len(subset),
                "input_above_days": in_above.sum(), "error_ready_days": ready.sum(),
                "error_above_days": err_above.sum(), "both_above_days": (in_above & err_above).sum(),
                "input_above_error_below_days": (in_above & ready & ~err_above).sum(),
                "input_alarms": subset.input_alarm.sum(), "error_alarms": subset.error_alarm.sum(),
                "first_input_alarm": subset.index[subset.input_alarm].min(),
                "first_error_alarm": subset.index[subset.error_alarm].min(),
                "budget_exhausted_alarms": subset.action.eq("budget_exhausted").sum(),
            })
        if strategy == "frozen":
            for day, row in group.loc[group.input_alarm | group.error_alarm].iterrows():
                before = group.loc[day - 27 * DAY:day, "daily_error_pct"]
                after = group.loc[day + DAY:day + 28 * DAY, "daily_error_pct"]
                alarm_windows.append({
                    "task": task, "market_day": day, "alarm_sources": row.alarm_sources,
                    "trailing_28d_error_pct": before.mean() if len(before) == 28 else np.nan,
                    "next_28d_error_pct": after.mean() if len(after) == 28 else np.nan,
                    "error_threshold_pct": row.error_threshold_pct,
                })
    return pd.DataFrame(rows), pd.DataFrame(alarm_windows)


def event_windows(frame):
    events = {
        "five_minute_settlement_start": pd.Timestamp("2021-10-01"),
        "nem_market_suspension_start": pd.Timestamp("2022-06-15"),
    }
    rows = []
    for label, day in events.items():
        for start, end, period in ((day - 28 * DAY, day - DAY, "prior_28d"),
                                   (day, day + 27 * DAY, "following_28d")):
            group = frame.loc[frame.market_day.between(start, end)]
            for method in METHODS:
                negative = group.actual.eq(0)
                positive = ~negative
                rows.append({
                    "event": label, "period": period, "first_day": start, "last_day": end,
                    "method": method, "actual_up_pct": 100 * positive.mean(),
                    "recall_0_pct": 100 * group.loc[negative, method].eq(0).mean(),
                    "recall_1_pct": 100 * group.loc[positive, method].eq(1).mean(),
                    **metric("price", group.actual, group[method]),
                })
    return pd.DataFrame(rows)


def plot_performance(task, monthly, yearly, updates, out):
    score = MONTHLY_SCORE[task]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.8), sharex=True, layout="constrained")
    for ax, strategy in zip(axes, STRATEGIES):
        values = monthly.loc[monthly.method.eq(strategy)]
        month = pd.to_datetime(values.period)
        color = COLORS[strategy]
        ax.fill_between(month, values.day_bootstrap_lower, values.day_bootstrap_upper,
                        color=color, alpha=.17, label="95% whole-day bootstrap")
        ax.fill_between(month, values.iid_lower, values.iid_upper, color=color, alpha=.30,
                        label="95% half-hour normal")
        ax.plot(month, values[score], color=color, label=strategy, linewidth=1.3)
        for baseline, style in zip(BASELINES, ("--", ":")):
            base = monthly.loc[monthly.method.eq(baseline)]
            ax.plot(month, base[score], color=COLORS[baseline], linestyle=style, linewidth=.9,
                    label=baseline.replace("_", " "))
        dates = updates.loc[updates.strategy.eq(strategy), "effective_day"]
        for i, day in enumerate(dates):
            ax.axvline(day, color=color, alpha=.55, lw=.8, label="Update effective" if i == 0 else None)
        ax.axvspan(SPLIT, END, color="#442463", alpha=.045)
        ax.set(ylabel=f"{score} (%)", title=strategy.capitalize())
        ax.grid(alpha=.16)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    for ax in axes:
        ax.set_ylim(min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes))
    axes[-1].set_xlabel("Market month (AEST) · shaded period from 2025 is the holdout")
    fig.suptitle(f"NSW {task}: monthly performance and update dates")
    fig.savefig(out / f"{task}_monthly.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, score in zip(axes, ("MAPE", "MAE") if task == "demand" else ("accuracy", "balanced_accuracy")):
        for method, group in yearly.groupby("method", sort=False):
            ax.plot(group.period.astype(int), group[score], label=method.replace("_", " "),
                    color=COLORS[method], marker="o", ms=3,
                    linestyle="--" if method in BASELINES else "-")
        ax.axvspan(2024.5, 2026.2, color="#442463", alpha=.055)
        ax.set_xlabel("Market year (2026: Jan–Oct)")
        unit = "MW" if score == "MAE" else "%"
        ax.set_ylabel(f"{score.replace('_', ' ')} ({unit})")
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"NSW {task}: yearly strategy and baseline comparison")
    fig.savefig(out / f"{task}_yearly.png", dpi=170)
    plt.close(fig)


def plot_detectors(trace, out):
    frozen = trace.loc[trace.strategy.eq("frozen")]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True, layout="constrained")
    series = (("demand", "input_rolling", "input_threshold", "input_alarm", "Day-type distance"),
              ("demand", "error_rolling_pct", "error_threshold_pct", "error_alarm", "Demand MAPE (%)"),
              ("price", "error_rolling_pct", "error_threshold_pct", "error_alarm", "Price error (%)"))
    for ax, (task, stat, threshold, alarm, label) in zip(axes, series):
        data = frozen.loc[frozen.task.eq(task)]
        ax.plot(data.market_day, data[stat], color="#006f70", lw=1.0, label="28-day mean")
        ax.step(data.market_day, data[threshold], where="post", color="#573b77", lw=.9,
                linestyle="--", label="2014 seasonal threshold")
        hits = data.loc[data[alarm]]
        ax.scatter(hits.market_day, hits[stat], color="#b84601", s=16, zorder=3, label="Alarm")
        ax.set_ylabel(label)
        ax.grid(alpha=.18)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Market day (AEST)")
    fig.suptitle("Online alarms on the frozen reference models")
    fig.savefig(out / "frozen_detector_timeline.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output
    updates = pd.read_csv(out / "retraining.csv", parse_dates=["effective_day"])
    trace = pd.read_csv(out / "detector_daily.csv", parse_dates=["market_day"])
    rng = np.random.default_rng(9417)
    pooled, monthly, yearly, contrasts = [], [], [], []
    for task in TASKS:
        frame = pd.read_csv(out / f"{task}_predictions.csv.gz", parse_dates=["market_day"])
        task_updates = updates.loc[updates.task.eq(task)]
        months, years, summary = score_periods(task, frame, frame.market_day, task_updates, rng)
        monthly.extend(months)
        yearly.extend(years)
        pooled.extend(summary)
        contrasts.extend(paired_effects(task, frame, rng))
        plot_performance(task, pd.DataFrame(months), pd.DataFrame(years), task_updates, out)
        if task == "price":
            event_windows(frame).to_csv(out / "market_event_windows.csv", index=False)

    pd.DataFrame(monthly).to_csv(out / "monthly_metrics.csv", index=False)
    pd.DataFrame(yearly).to_csv(out / "yearly_metrics.csv", index=False)
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(out / "summary.csv", index=False)
    select_strategy(pooled).to_csv(out / "selection.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_gains.csv", index=False)
    rates, windows = detector_evidence(trace)
    rates.to_csv(out / "detector_summary.csv", index=False)
    windows.to_csv(out / "frozen_alarm_windows.csv", index=False)
    plot_detectors(trace, out)


if __name__ == "__main__":
    main()
