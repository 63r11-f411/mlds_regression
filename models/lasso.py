import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from preprocessing import build_scaled_preprocessor
from config import RANDOM_STATE


PARAM_GRID = [
    {"lasso__alpha": np.logspace(-5, -1, 6)}
]

def get_lasso_pipeline(preprocessor=None):
    if preprocessor is None:
        preprocessor = build_scaled_preprocessor()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        # при малых alpha покоординатному спуску не хватает дефолтных 1000 итераций
        ('lasso', Lasso(max_iter=20000, random_state=RANDOM_STATE))
    ])
    return pipeline