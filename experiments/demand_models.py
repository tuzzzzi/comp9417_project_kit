"""Fresh sklearn estimators for A; predict() always returns demand in MW.

Copy this module AND demand_config.json when integrating with the group repository.
The JSON records choices made on 2014; fit() never reads the validation set.
"""

import json
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONFIG_PATH = Path(__file__).with_name("demand_config.json")
DEFAULT_MLP = {
    "hidden_layer_sizes": [64, 32],
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 256,
    "learning_rate_init": 0.001,
    "max_iter": 300,
    "early_stopping": False,
    "shuffle": True,
    "n_iter_no_change": 30,
    "tol": 0.00001,
    "random_state": 42,
}


def make_ridge(alpha=10.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def make_mlp(**overrides):
    settings = {**DEFAULT_MLP, **overrides}
    settings["hidden_layer_sizes"] = tuple(settings["hidden_layer_sizes"])
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(**settings)),
        transformer=StandardScaler(),
    )


def build_models():
    """Return new, unfitted models, suitable for initial fitting or retraining."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Selected demand configuration missing: {CONFIG_PATH}. "
            "Copy demand_config.json together with demand_models.py, or run "
            "experiments.run_demand_study to select a configuration."
        )
    config = json.loads(CONFIG_PATH.read_text())
    if config.get("schema_version") != 1 or "ridge_alpha" not in config or not isinstance(config.get("mlp_params"), dict):
        raise ValueError("Incomplete or unsupported demand_config.json; use the selected study configuration.")
    missing = set(DEFAULT_MLP) - set(config["mlp_params"])
    if missing:
        raise ValueError(f"demand_config.json is missing MLP settings: {sorted(missing)}")
    return {
        "ridge": make_ridge(config["ridge_alpha"]),
        "mlp": make_mlp(**config["mlp_params"]),
    }
