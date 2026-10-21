from sklearn.pipeline import Pipeline
from preprocessing import build_scaled_preprocessor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from config import RANDOM_STATE

PARAM_GRID = {
    'knn__n_neighbors': list(range(1, 22, 2)),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'cosine'],
    'feature_selection__max_features': [5, 10, 15, 20]
}

ALPHA = 0.0004  #около оптимальное значение из модели лассо

def get_knn_pipeline(preprocessor=None):
    if preprocessor is None:
        preprocessor = build_scaled_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('feature_selection', SelectFromModel(Lasso(alpha=ALPHA, max_iter=20000, random_state=RANDOM_STATE))),
        ('knn', KNeighborsRegressor())
    ])
