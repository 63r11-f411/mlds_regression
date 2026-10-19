import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from preprocessing import build_scaled_preprocessor
from config import RANDOM_STATE


PARAM_GRID = [
    {"ridge__alpha": np.logspace(-3, 2, 6)}
]

def get_ridge_pipeline(preprocessor=None):
    if preprocessor is None:
        preprocessor = build_scaled_preprocessor()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ridge', Ridge(random_state=RANDOM_STATE))
    ])
    return pipeline