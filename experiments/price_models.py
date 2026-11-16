"""Model factories owned by the price direction group."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_models():
    """Return fresh, unfitted estimators whose predict method returns 0 or 1."""
    return {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500)
        ),
        # Add "hist_gbc" here once the classification group has chosen its settings.
    }
