"""Model factories owned by the price direction group."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_models():
    """Return fresh, unfitted estimators whose predict method returns 0 or 1."""
    return {
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500)
        ),
        "hist_gbc": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            class_weight="balanced",
            # Keep all training years; avoid a random early-stopping split.
            early_stopping=False,
            random_state=42,
        ),
    }
