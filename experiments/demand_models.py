"""Model factories owned by the demand regression group."""

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_models():
    """Return fresh, unfitted regression estimators keyed by model name."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        # Add "mlp" here once the regression group has chosen its training settings.
    }
