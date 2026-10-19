import datetime
import json
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from config import TARGET, RANDOM_STATE, METRIC, CV_FOLDS, SUBMISSION_DIR, BEST_PARAMS_DIR, SCORE_LOG


def make_cv():
    """Возвращает объект KFold для кросс-валидации"""
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def build_estimator(model, param_grid=None, n_iter=None, cv=None, n_jobs=-1):
    """Возвращает модель с настроенным поиском гиперпараметров (если параметры заданы)"""
    if param_grid is None:
        return model
    if cv is None:
        cv = make_cv()
    if n_iter is not None:
        return RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv, scoring=METRIC, random_state=RANDOM_STATE, n_jobs=n_jobs)
    return GridSearchCV(model, param_grid, cv=cv, scoring=METRIC, n_jobs=n_jobs)


def evaluate_model(model, X, y, cv=None, n_jobs=-1, param_grid=None, n_iter=None):
    """Оценивает модель с помощью кросс-валидации"""
    if cv is None:
        cv = make_cv()
    estimator = build_estimator(model, param_grid, n_iter, cv, n_jobs)
    outer_n_jobs = n_jobs if param_grid is None else None
    return cross_val_score(estimator, X, y, cv=cv, scoring=METRIC, n_jobs=outer_n_jobs)


def save_submission(house_ids, predictions, name):
    """Сохраняет предсказания модели в файл для отправки на Kaggle"""
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / f"{name}.csv"
    submission = pd.DataFrame({'Id': house_ids, TARGET: predictions.astype(float)})
    submission.to_csv(path, index=False)
    return path

def save_best_params(name, params):
    """Сохраняет лучшие параметры модели"""
    BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = BEST_PARAMS_DIR / f"{name}.json"
    path.write_text(json.dumps(params, indent=2, default=lambda value: value.item()))
    return path


def load_best_params(name):
    """Возвращает сохраненные лучшие параметры модели, или пустой словарь, если они не были сохранены."""
    path = BEST_PARAMS_DIR / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def log_scores(name, scores, params, tuned):
    """Записывает результаты оценки модели в лог"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(SCORE_LOG, 'a') as score_file:
        score_file.write(f"{timestamp} | {name} | {'Param tuned' if tuned else 'No param tuning'} | {METRIC}\n")
        score_file.write(f'{scores.mean():.4f} ± {scores.std():.4f}\n')
        score_file.write(f'{scores}\n')
        if params:
            score_file.write(f'params: {params}\n')
        score_file.write('\n')
