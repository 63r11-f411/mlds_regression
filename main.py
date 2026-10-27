import pandas as pd
import numpy as np
from config import RUNCONF, DATA_DIR, TARGET, METRIC, DROP_TRAIN_OUTLIERS
from models import MODELS, build_model, resolve_params
from utility import build_estimator, evaluate_model, save_submission, save_best_params, log_scores



if __name__ == "__main__":
    train = pd.read_csv(DATA_DIR / "train.csv")
    train.drop("Id", axis=1, inplace=True)

    if DROP_TRAIN_OUTLIERS: # выбросы, портили CV: два больших дома с аномально низкой ценой
        train = train[~((train['GrLivArea'] > 4000) & (np.log1p(train['SalePrice']) < 12.5))]

    X, y = train.drop(columns=[TARGET]), train[TARGET]
    y = np.log1p(y) # метрика соревнования — RMSE логарифмов
    
    X_test = pd.read_csv(DATA_DIR / "test.csv")
    test_ids = X_test.pop("Id")

    for model_type in RUNCONF.models:
        print(f'=== {model_type} ===')
        model_info = MODELS[model_type]
        model = build_model(model_type)
        param_grid = model_info.param_grid if RUNCONF.tune_hyperparameters else None
        search_params = model_info.search_params
        params = resolve_params(model_type)

        # подбираем лучшие параметры на всём наборе данных или обучаем для сабмита
        if param_grid is not None or RUNCONF.save_test_results:
            estimator = build_estimator(model, param_grid=param_grid, **search_params)
            estimator.fit(X, y)
            if param_grid is not None:
                params = estimator.best_params_
                path = save_best_params(model_type, params)
                print(f'best params: {params}, saved to {path}')
            if RUNCONF.save_test_results:
                pred = estimator.predict(X_test)
                pred = np.expm1(pred)
                path = save_submission(test_ids, pred, model_type)
                print(f'submission saved to {path}')

        
        if RUNCONF.cross_validate:
            scores = evaluate_model(model, X, y, param_grid=param_grid, **search_params)
            print(f'scores: {scores}')
            print(f'mean {METRIC}: {scores.mean():.4f}')
            print(f'std: {scores.std():.4f}')
            log_scores(model_type, scores, params, tuned=param_grid is not None)
