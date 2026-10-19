from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from preprocessing import build_unscaled_preprocessor
from config import RANDOM_STATE

PARAM_GRID = {
    'rf__n_estimators': [300],
    'rf__max_depth': [4, 6, 8, None],
    'rf__min_samples_leaf': [1, 2, 5, 10],
    'rf__max_features': ['sqrt', 0.5, None],
}

def get_rf_pipeline(preprocessor=None):
    if preprocessor is None:
        preprocessor = build_unscaled_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('rf', RandomForestClassifier(max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE))
    ])
