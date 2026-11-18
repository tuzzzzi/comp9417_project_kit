"""Export static report figures and 2014 descriptive error tables for A."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stage1_data import ROOT

COLORS = {"yesterday": "#777777", "last_week": "#B2B2B2",
          "ridge": "#0072B2", "mlp": "#D55E00"}
LABELS = {"yesterday": "Yesterday", "last_week": "Last week", "ridge": "Ridge", "mlp": "MLP"}


def save(fig, folder, name, svg=True):
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    if svg:
        fig.savefig(folder / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def monthly_intervals(long):
    """IID normal interval and whole-day percentile bootstrap (2,000 replicates)."""
    records = []
    rng = np.random.default_rng(42)
    for (model, month), group in long.groupby(["model", "month"], sort=True):
        daily = group.groupby("market_day").ape.mean().to_numpy()
        draws = rng.integers(0, len(daily), size=(2000, len(daily)))
        boot = daily[draws].mean(axis=1)
        mean = group.ape.mean()
        half_width = 1.96 * group.ape.std(ddof=1) / np.sqrt(len(group))
        low, high = np.percentile(boot, [2.5, 97.5])
        records.append({"model": model, "month": str(month), "n_halfhours": len(group),
                        "n_days": len(daily), "MAPE": mean, "MAE": group.ae.mean(),
                        "iid_lower": mean - half_width, "iid_upper": mean + half_width,
                        "day_bootstrap_lower": low, "day_bootstrap_upper": high,
                        "bootstrap_replicates": 2000, "bootstrap_seed": 42})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/stage1")
    args = parser.parse_args()
    output = args.input
    folder = output / "demand_figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "axes.axisbelow": True,
                         "svg.fonttype": "none"})
    scores = pd.read_csv(output / "demand_validation.csv")
    predictions = pd.read_csv(output / "demand_predictions.csv", parse_dates=["timestamp", "market_day"])
    long = pd.read_csv(output / "demand_predictions_long.csv", parse_dates=["timestamp", "market_day"])
    long["ape"] = 100 * (long.y_demand - long.pred_demand).abs() / long.y_demand
    long["ae"] = (long.y_demand - long.pred_demand).abs()
    long["error_mw"] = long.pred_demand - long.y_demand
    long["month"] = long.market_day.dt.to_period("M")
    long["slot"] = (long.timestamp.dt.hour * 2 + long.timestamp.dt.minute // 30 - 1) % 48
    long["day_type"] = np.where(long.market_day.dt.day_of_week >= 5, "Weekend", "Weekday")
    for keys, name in [(["model", "market_day"], "daily"),
                       (["model", "slot"], "by_slot"),
                       (["model", "day_type"], "by_weekday")]:
        long.groupby(keys).agg(n_halfhours=("ape", "size"), MAPE=("ape", "mean"),
                               MAE=("ae", "mean"), bias_mw=("error_mw", "mean")).reset_index().to_csv(
            output / f"demand_{name}.csv", index=False)
    monthly = monthly_intervals(long)
    monthly.to_csv(output / "demand_monthly.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    positions = np.arange(len(scores))
    for ax, metric, unit in zip(axes, ["MAPE", "MAE"], ["%", "MW"]):
        bars = ax.bar(positions, scores[metric], color=[COLORS[m] for m in scores.model])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xticks(positions, [LABELS[m] for m in scores.model])
        ax.set_ylabel(f"{metric} ({unit}); lower is better")
        ax.set_ylim(0, scores[metric].max() * 1.2)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Demand forecasting: 2014 model-selection validation")
    save(fig, folder, "validation_scores")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    for ax, start in zip(axes, [pd.Timestamp("2014-01-01"), pd.Timestamp("2014-07-01")]):
        week = predictions[(predictions.market_day >= start) &
                           (predictions.market_day < start + pd.Timedelta(days=7))]
        ax.plot(week.timestamp, week.actual, color="#222222", label="Observed", linewidth=1.3)
        for model, style in [("ridge", "--"), ("mlp", "-")]:
            ax.plot(week.timestamp, week[model], color=COLORS[model], linestyle=style,
                    linewidth=1, alpha=0.9, label=LABELS[model])
        ax.set_title(f"Preselected week: {start.date()} to {(start + pd.Timedelta(days=6)).date()}")
        ax.set_ylabel("NSW demand (MW)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Interval end in fixed AEST (UTC+10)")
    save(fig, folder, "seasonal_week_predictions")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ["ridge", "mlp"]):
        m = monthly[monthly.model == model].sort_values("month")
        x = np.arange(1, 13)
        ax.fill_between(x, m.day_bootstrap_lower, m.day_bootstrap_upper,
                        color=COLORS[model], alpha=0.2, label="95% whole-day bootstrap")
        ax.errorbar(x, m.MAPE, yerr=[m.MAPE - m.iid_lower, m.iid_upper - m.MAPE],
                    color=COLORS[model], marker="o", capsize=3, label="Mean + 95% IID normal CI")
        baseline = monthly[monthly.model == "yesterday"].sort_values("month")
        ax.plot(x, baseline.MAPE, color=COLORS["yesterday"], linestyle="--", label="Yesterday mean")
        ax.set_title(LABELS[model])
        ax.set_ylabel("Monthly MAPE (%)")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    axes[-1].set_xticks(np.arange(1, 13))
    axes[-1].set_xlabel("Month in 2014 (used for model selection)")
    save(fig, folder, "monthly_mape_intervals")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    lo = min(predictions.actual.min(), predictions[["ridge", "mlp"]].min().min())
    hi = max(predictions.actual.max(), predictions[["ridge", "mlp"]].max().max())
    margin = (hi - lo) * 0.025
    lo, hi = lo - margin, hi + margin
    for ax, model in zip(axes, ["ridge", "mlp"]):
        ax.scatter(predictions.actual, predictions[model], s=3, alpha=0.12, color=COLORS[model], rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Observed demand (MW)", ylabel="Predicted demand (MW)",
               title=f"{LABELS[model]}: all 17,520 validation intervals", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_aspect("equal", adjustable="box")
    save(fig, folder, "observed_vs_predicted", svg=False)

    config = json.loads((output / "demand_model_config.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    tuning = pd.read_csv(output / "demand_tuning.csv")
    for name in tuning.loc[tuning.family == "mlp", "candidate"]:
        curve = pd.read_csv(output / "loss_curves" / f"{name}.csv")
        selected = name == config["selected_candidates"]["mlp"]
        ax.plot(curve.epoch, curve.scaled_training_loss, label=name + (" (selected)" if selected else ""),
                linewidth=2 if selected else 1, linestyle="-" if selected else "--")
    ax.set(xlabel="Training epoch", ylabel="Training loss on standardized demand",
           title="MLP optimization diagnostics (not validation MAPE)")
    ax.set_yscale("log")
    ax.legend()
    save(fig, folder, "training_loss")

    ablations = pd.read_csv(output / "demand_feature_ablation.csv")
    groups = ["none", "weather", "calendar", "demand_lags", "price_history"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(groups))
    for shift, model in [(-0.18, "ridge"), (0.18, "mlp")]:
        values = ablations[ablations.model == model].set_index("removed_group").loc[groups, "MAPE"]
        bars = ax.bar(x + shift, values, width=0.36, label=LABELS[model], color=COLORS[model])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x, ["All 22", "Without weather", "Without calendar", "Without demand lags", "Without price history"], rotation=12)
    ax.set(ylabel="2014 MAPE (%); lower is better", title="Feature-group ablation with fixed selected hyperparameters")
    ax.set_ylim(0, ablations.MAPE.max() * 1.18)
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, folder, "feature_ablation")

    grouped = long.groupby(["model", "slot"]).ape.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    for model, style in [("yesterday", ":"), ("last_week", "-."), ("ridge", "--"), ("mlp", "-")]:
        ax.plot(grouped.index, grouped[model], color=COLORS[model], label=LABELS[model], linestyle=style)
    ax.set(xlabel="Half-hour slot (0 = interval ending 00:30; 47 = next midnight)",
           ylabel="2014 MAPE (%)", title="Error by market-day half-hour")
    ax.legend(ncol=4)
    save(fig, folder, "error_by_slot")
    print(f"Saved seven PNG figures and companion tables to {output}")


if __name__ == "__main__":
    main()
