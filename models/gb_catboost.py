from catboost import CatBoostRegressor
from scipy.stats import loguniform, randint
from sklearn.pipeline import Pipeline
from preprocessing import build_native_categorical_preprocessor, NATIVE_CATEGORICAL_FEATURES
from config import RANDOM_STATE

PARAM_GRID = {
    'catboost__iterations': randint(500, 3000),
    'catboost__depth': [4, 5, 6],
    'catboost__l2_leaf_reg': loguniform(0.1, 10),
}

def get_catboost_pipeline(preprocessor=None):
    '''
    Создает пайплайн для CatBoost регрессора с предварительной обработкой данных. Использует нативные категории
    '''
    if preprocessor is None:
        preprocessor = build_native_categorical_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('catboost', CatBoostRegressor(
            cat_features=tuple(NATIVE_CATEGORICAL_FEATURES), random_state=RANDOM_STATE, verbose=False
        ))
    ])
