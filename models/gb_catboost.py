from catboost import CatBoostClassifier
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from preprocessing import build_native_categorical_preprocessor, NATIVE_CATEGORICAL_FEATURES
from config import RANDOM_STATE

PARAM_GRID = {
    'catboost__iterations': [300, 600, 1000],
    'catboost__learning_rate': loguniform(0.01, 0.2),
    'catboost__depth': [4, 6, 8],
    'catboost__l2_leaf_reg': [1, 3, 5, 10],
}

def get_catboost_pipeline(preprocessor=None):
    if preprocessor is None:
        preprocessor = build_native_categorical_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('catboost', CatBoostClassifier(
            cat_features=tuple(NATIVE_CATEGORICAL_FEATURES), random_state=RANDOM_STATE, verbose=False
        ))
    ])
