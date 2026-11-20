"""Selected demand architecture; its settings were chosen in Stage 1."""
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(params):
    params = dict(params)
    params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), MLPRegressor(**params)),
        transformer=StandardScaler(),
    )
