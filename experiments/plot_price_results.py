"""Plot the price results produced by experiments.validate_models."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .stage1_data import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "stage1")
    args = parser.parse_args()
    scores = pd.read_csv(args.input / "price_validation.csv")
    predictions = pd.read_csv(
        args.input / "price_predictions.csv", parse_dates=["timestamp", "market_day"]
    )
    figures = args.input / "price_figures"
    figures.mkdir(parents=True, exist_ok=True)
    models = scores["model"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(models))
    ax.bar(positions - 0.18, scores["accuracy"], width=0.36, label="Accuracy")
    ax.bar(positions + 0.18, scores["balanced_accuracy"], width=0.36,
           label="Balanced accuracy")
    ax.set_xticks(positions, models)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Price direction: 2014 validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "validation_scores.png", dpi=200)
    plt.close(fig)

    # Use the first full validation week, selected without looking at its errors.
    start = predictions["market_day"].min()
    week = predictions[predictions["market_day"] < start + pd.Timedelta(days=7)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, model in zip(axes, ["logistic", "hist_gbc"]):
        ax.step(week["timestamp"], week["actual"], where="post",
                label="Actual", color="black", linewidth=1)
        ax.step(week["timestamp"], week[model], where="post",
                label=model, alpha=0.7, linewidth=1)
        ax.set_yticks([0, 1])
        ax.set_ylabel("Price class")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Interval end (AEST)")
    fig.suptitle("First validation week: price above previous day's mean")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / "first_week_predictions.png", dpi=200)
    plt.close(fig)

    rows = []
    for model in models:
        tn, fp, fn, tp = confusion_matrix(
            predictions["actual"], predictions[model], labels=[0, 1]
        ).ravel()
        rows.append({
            "model": model,
            "actual_positive_pct": predictions["actual"].mean() * 100,
            "predicted_positive_pct": predictions[model].mean() * 100,
            "recall_0_pct": tn / (tn + fp) * 100,
            "recall_1_pct": tp / (tp + fn) * 100,
            "true_negative": tn, "false_positive": fp,
            "false_negative": fn, "true_positive": tp,
        })
    pd.DataFrame(rows).to_csv(args.input / "price_class_balance.csv", index=False)
    print(f"Saved figures to {figures}")


if __name__ == "__main__":
    main()
