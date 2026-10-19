from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from preprocessing import build_native_categorical_preprocessor
from config import RANDOM_STATE
import lightgbm as lgb

PARAM_GRID = {
    'lgb__n_estimators': [300, 600, 1000],
    'lgb__learning_rate': loguniform(0.01, 0.2),
    'lgb__num_leaves': [3, 7, 15],
}

def get_lgb_pipeline(preprocessor=None, num_leaves=7, verbose=0, **lgb_params):
    if preprocessor is None:
        preprocessor = build_native_categorical_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('lgb', lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=verbose, num_leaves=num_leaves, **lgb_params))
    ])
