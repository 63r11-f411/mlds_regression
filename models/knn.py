from sklearn.pipeline import Pipeline
from preprocessing import build_scaled_preprocessor
from sklearn.neighbors import KNeighborsClassifier

PARAM_GRID = {
    'knn__n_neighbors': list(range(1, 22, 2)),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'cosine'],
}

def get_knn_pipeline(preprocessor=None):
    if preprocessor is None:
        preprocessor = build_scaled_preprocessor()
    return Pipeline([
        ('preprocessing', preprocessor),
        ('knn', KNeighborsClassifier())
    ])
