"""Shared day-ahead inputs, targets, and baseline predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

from harness import dayahead as da

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv"
VALIDATION_START = pd.Timestamp("2014-01-01")
VALIDATION_END = pd.Timestamp("2015-01-01")
TARGET_COLUMNS = {"demand": "nsw_demand_mw", "price": "nsw_price_up"}
BASELINE_LAGS = {"yesterday": 1, "last_week": 7}


def build_features(history, weather):
    """Create features for the timestamps in weather using earlier market days."""
    index = weather.index
    days = da.market_day(index)
    slot = (index.hour * 2 + index.minute // 30 - 1) % da.N_PERIODS
    features = weather.loc[:, da.WEATHER_COLS].copy()
    features["slot_sin"] = np.sin(2 * np.pi * slot / da.N_PERIODS)
    features["slot_cos"] = np.cos(2 * np.pi * slot / da.N_PERIODS)
    features["weekend"] = (days.dayofweek >= 5).astype(int)
    features["season_sin"] = np.sin(2 * np.pi * (days.dayofyear - 1) / 365.25)
    features["season_cos"] = np.cos(2 * np.pi * (days.dayofyear - 1) / 365.25)

    market = history.loc[:, ["nsw_demand", "nsw_price"]]
    for lag in BASELINE_LAGS.values():
        previous = market.reindex(index - pd.Timedelta(days=lag))
        features[f"demand_lag{lag}"] = previous["nsw_demand"].to_numpy()
        features[f"price_lag{lag}"] = previous["nsw_price"].to_numpy()

    # Use the same reduction as the marking harness to preserve ties at the threshold.
    daily_mean = market["nsw_price"].groupby(da.market_day(market.index)).agg(
        lambda prices: prices.mean()
    )
    features["previous_price_mean"] = daily_mean.reindex(
        days - pd.Timedelta(days=1)
    ).to_numpy()
    return features


def load_stage1(csv_path):
    """Return separate features and labels through the final market day of 2014."""
    data = da.load_dataset(csv_path)
    data = data.loc[da.market_day(data.index) < VALIDATION_END]
    features = build_features(data, data.loc[:, da.WEATHER_COLS])
    # A full week of history is needed to compare against the lag-7 baseline.
    days = da.market_day(features.index)
    first_forecast_day = days[0] + pd.Timedelta(days=max(BASELINE_LAGS.values()))
    features = features.loc[days >= first_forecast_day]
    observed = data.loc[features.index]
    truth = pd.DataFrame({
        TARGET_COLUMNS["demand"]: observed["nsw_demand"],
        TARGET_COLUMNS["price"]: (observed["nsw_price"] > features["previous_price_mean"]).astype(int),
    })
    return features, truth


def baseline_predictions(features, task):
    """Use lag 1 and lag 7 with the official previous-day price threshold."""
    if task == "demand":
        return {name: features[f"demand_lag{lag}"] for name, lag in BASELINE_LAGS.items()}
    return {
        name: (features[f"price_lag{lag}"] > features["previous_price_mean"]).astype(int)
        for name, lag in BASELINE_LAGS.items()
    }
