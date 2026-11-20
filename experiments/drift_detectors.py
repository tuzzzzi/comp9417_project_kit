"""Seasonal online drift monitoring for the Stage 2 backtest."""
from collections import deque

import numpy as np
import pandas as pd

WINDOW = 28
PERSISTENCE = 7
QUANTILE = 0.95
SEASONS = ("summer", "autumn", "winter", "spring")


def season(day):
    return SEASONS[(day.month % 12) // 3]


def calibrate(reference):
    """Use full windows whose observations are all in the 2014 reference year."""
    rolling = reference.rolling(WINDOW).mean().dropna()
    grouped = rolling.groupby(rolling.index.map(season))
    thresholds = grouped.quantile(QUANTILE).reindex(SEASONS)
    if thresholds.isna().any().any():
        raise ValueError("Every season needs complete calibration windows.")
    return thresholds, grouped.size().reindex(SEASONS)


class Detector:
    """Emit one alarm per uninterrupted run above the seasonal threshold."""

    def __init__(self, thresholds, history):
        self.thresholds = thresholds
        self.values = deque(history, maxlen=WINDOW)
        self.streak = 0

    def reset(self):
        self.values.clear()
        self.streak = 0

    def observe(self, day, value):
        self.values.append(float(value))
        threshold = self.thresholds[season(day)]
        if len(self.values) < WINDOW:
            return np.nan, threshold, False
        statistic = float(np.mean(self.values))
        self.streak = self.streak + 1 if statistic > threshold else 0
        return statistic, threshold, self.streak == PERSISTENCE
