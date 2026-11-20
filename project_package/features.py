"""Shared causal features for final training and daily inference."""
import numpy as np
import pandas as pd

WEATHER_COLS = [f'{city}_{v}' for city in ('syd', 'mel', 'bne')
                for v in ('temp', 'rh', 'cloud', 'solar')]


def market_day(index):
    return (index - pd.Timedelta('30min')).normalize()


def build_features(history, weather):
    index = weather.index
    days = market_day(index)
    slot = (index.hour * 2 + index.minute // 30 - 1) % 48
    features = weather.loc[:, WEATHER_COLS].copy()
    features['slot_sin'] = np.sin(2 * np.pi * slot / 48)
    features['slot_cos'] = np.cos(2 * np.pi * slot / 48)
    features['weekend'] = (days.day_of_week >= 5).astype(int)
    features['season_sin'] = np.sin(2 * np.pi * (days.dayofyear - 1) / 365.25)
    features['season_cos'] = np.cos(2 * np.pi * (days.dayofyear - 1) / 365.25)
    market = history.loc[:, ['nsw_demand', 'nsw_price']]
    for lag in (1, 7):
        previous = market.reindex(index - pd.Timedelta(days=lag))
        features[f'demand_lag{lag}'] = previous.nsw_demand.to_numpy()
        features[f'price_lag{lag}'] = previous.nsw_price.to_numpy()
    # Preserve the official strict-comparison label, including floating-point ties.
    means = market.nsw_price.groupby(market_day(market.index)).agg(lambda p: p.mean())
    features['previous_price_mean'] = means.reindex(days-pd.Timedelta(days=1)).to_numpy()
    return features
