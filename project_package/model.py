"""Single marking interface for both tasks. No training or persistent day state."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from features import WEATHER_COLS, build_features, market_day


class Forecaster:
    def __init__(self, artifact_dir: str):
        folder = Path(artifact_dir)
        self.config = json.loads((folder / 'training.json').read_text())
        self.demand = joblib.load(folder / 'demand.joblib')
        self.price = joblib.load(folder / 'price.joblib')

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        index = weather_next.index
        if len(index) != 48 or not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError('Expected 48 ordered, unique target intervals')
        day = market_day(index)[0]
        expected = pd.date_range(day + pd.Timedelta('30min'), periods=48, freq='30min')
        if not index.equals(expected):
            raise ValueError('Target intervals must cover one complete market day')
        # Filter before taking the tail: no target-day market observations are used.
        past = history.loc[market_day(history.index) < day].sort_index().tail(7*48)
        expected_past = pd.date_range(day-pd.Timedelta(days=7)+pd.Timedelta('30min'),
                                      periods=7*48, freq='30min')
        if not past.index.equals(expected_past):
            raise ValueError('Expected seven complete preceding market days')
        x = build_features(past, weather_next.loc[:, WEATHER_COLS])
        if list(x.columns) != self.config['feature_names'] or not np.isfinite(x.to_numpy()).all():
            raise ValueError('Features must match training and contain finite values')
        with threadpool_limits(limits=2):
            demand = np.asarray(self.demand.predict(x), dtype=float)
            price = np.asarray(self.price.predict(x))
        if demand.shape != (48,) or price.shape != (48,) or not np.isfinite(demand).all():
            raise ValueError('Invalid model predictions')
        if not np.isin(price, [0, 1]).all():
            raise ValueError('Price outputs must be classes 0 or 1')
        return pd.DataFrame({'nsw_demand_mw': np.clip(demand, 0, 20000),
                             'nsw_price_up': price.astype(int)}, index=index)
