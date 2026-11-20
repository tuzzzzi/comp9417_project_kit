"""Selected price-direction architecture from Stage 1."""
from sklearn.ensemble import HistGradientBoostingClassifier


def build_model(params):
    return HistGradientBoostingClassifier(**params)
