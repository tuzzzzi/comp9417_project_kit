"""Train on 2010-2013 and compare each task's models and baselines on 2014."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from harness import dayahead as da

from . import demand_models, price_models
from .stage1_data import (
    DATA_PATH,
    ROOT,
    TARGET_COLUMNS,
    VALIDATION_START,
    baseline_predictions,
    load_stage1,
)

MODEL_BUILDERS = {"demand": demand_models.build_models, "price": price_models.build_models}


def score(task, actual, predicted):
    if task == "demand":
        return {"MAPE": da.mape(actual, predicted), "MAE": da.mae(actual, predicted)}
    return {
        "accuracy": da.accuracy(actual, predicted),
        "balanced_accuracy": da.balanced_accuracy(actual, predicted),
    }


def evaluate_task(task, train_x, valid_x, train_y, valid_y):
    forecasts = baseline_predictions(valid_x, task)
    for name, estimator in MODEL_BUILDERS[task]().items():
        estimator.fit(train_x, train_y)
        predicted = np.asarray(estimator.predict(valid_x))
        if predicted.shape != (len(valid_x),) or not np.isfinite(predicted).all():
            raise ValueError(f"{name} must return one finite prediction per half-hour.")
        if task == "price" and not np.isin(predicted, [0, 1]).all():
            raise ValueError(f"{name} must return 0/1 class labels from predict().")
        forecasts[name] = predicted

    scores = pd.DataFrame([
        {"model": name, **score(task, valid_y, forecast)}
        for name, forecast in forecasts.items()
    ])
    predictions = pd.DataFrame({
        "market_day": da.market_day(valid_x.index),
        "actual": valid_y,
        **forecasts,
    }, index=valid_x.index)
    return scores, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*MODEL_BUILDERS, "both"], default="both")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stage1")
    args = parser.parse_args()

    features, truth = load_stage1(args.data)
    training = da.market_day(features.index) < VALIDATION_START
    train_x, valid_x = features.loc[training], features.loc[~training]
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = MODEL_BUILDERS if args.task == "both" else (args.task,)

    with threadpool_limits(limits=2):
        for task in tasks:
            labels = truth[TARGET_COLUMNS[task]]
            scores, predictions = evaluate_task(
                task, train_x, valid_x, labels.loc[training], labels.loc[~training]
            )
            scores.to_csv(args.output / f"{task}_validation.csv", index=False)
            predictions.to_csv(args.output / f"{task}_predictions.csv", index_label="timestamp")
            print(f"\n{task} - 2014 validation")
            print(scores.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
