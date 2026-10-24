import pathlib
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "ds"
TARGET = "SalePrice"
RANDOM_STATE = 42
METRIC = "neg_root_mean_squared_error"
CV_FOLDS = 5
SUBMISSION_DIR = ROOT / "submission"
BEST_PARAMS_DIR = ROOT / "best_params"
SCORE_LOG = ROOT / "scores.txt"
DROP_TRAIN_OUTLIERS = True

@dataclass(frozen=True)
class RunConfig:
    models: tuple[str, ...] = ('ridge', 'lasso', 'elasticnet', 'knn', 'tree')
    tune_hyperparameters: bool = True
    cross_validate: bool = True
    save_test_results: bool = True
    use_saved_params: bool = True
    voting_members: tuple[str, ...] = ()

RUNCONF = RunConfig()