from sklearn.pipeline import Pipeline
from preprocessing import build_unscaled_preprocessor
from config import RANDOM_STATE
from xgboost import XGBRegressor
from scipy.stats import loguniform, randint, uniform

PARAM_GRID = {
    'xgb__n_estimators': randint(300, 3000),
    'xgb__max_depth': [1, 2, 3, 4, 5],
    'xgb__min_child_weight': [1, 2, 3, 5],
    'xgb__subsample': uniform(0.5, 0.5),         # [0.5, 1.0]
    'xgb__colsample_bytree': uniform(0.3, 0.7),  # [0.3, 1.0]
    'xgb__reg_lambda': loguniform(0.1, 10),
}

def get_xgb_pipeline(preprocessor=None, max_depth=7, **xgb_params):
    if preprocessor is None:
        preprocessor = build_unscaled_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('xgb', XGBRegressor(random_state=RANDOM_STATE, max_depth=max_depth, **xgb_params))
    ])
