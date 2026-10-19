import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from preprocessing import build_scaled_preprocessor
from config import RANDOM_STATE


PARAM_GRID = [
    {"elasticnet__alpha": np.logspace(-4, 0, 5),
     "elasticnet__l1_ratio": np.linspace(0.1, 0.9, 9)}
]

def get_elasticnet_pipeline(preprocessor=None):
    if preprocessor is None:
        preprocessor = build_scaled_preprocessor()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        # при малых alpha покоординатному спуску не хватает дефолтных 1000 итераций
        ('elasticnet', ElasticNet(max_iter=20000, random_state=RANDOM_STATE))
    ])
    return pipeline