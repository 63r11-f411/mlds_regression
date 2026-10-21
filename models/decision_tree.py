from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from preprocessing import build_unscaled_preprocessor
from config import RANDOM_STATE

PARAM_GRID = {
    'tree__max_depth': [3, 4, 5, 6, 8, None],
    'tree__min_samples_leaf': [1, 2, 5, 10, 20],
    'tree__ccp_alpha': [0.0, 0.01, 0.1, 1.0]
}

def get_tree_pipeline(preprocessor=None):
    if preprocessor is None:
        preprocessor = build_unscaled_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('tree', DecisionTreeRegressor(random_state=RANDOM_STATE))
    ])
