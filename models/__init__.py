from collections.abc import Callable
from functools import partial
from dataclasses import dataclass, field
from config import RUNCONF
from utility import load_best_params
from .baseline import get_ridge_pipeline, PARAM_GRID as RIDGE_GRID
from .lasso import get_lasso_pipeline, PARAM_GRID as LASSO_GRID
from .elasticnet import get_elasticnet_pipeline, PARAM_GRID as ELASTICNET_GRID
from .knn import get_knn_pipeline, PARAM_GRID as KNN_GRID
from .decision_tree import get_tree_pipeline, PARAM_GRID as TREE_GRID
from .gb_xgb import get_xgb_pipeline, PARAM_GRID as XGB_GRID
from .gb_catboost import get_catboost_pipeline, PARAM_GRID as CATBOOST_GRID
from .nn import get_nn_pipeline, PARAM_GRID as NN_GRID
from .voting import get_voting_regressor
from .stacking import get_stacking_regressor

@dataclass(frozen=True)
class MLModelConf:
    """Конфигурация модели машинного обучения"""
    build: Callable
    param_grid: dict | list[dict] | None = None # используется для подбора оптимальных параметров
    model_params: dict = field(default_factory=dict) # напрямую задаёт параметры модели
    search_params: dict = field(default_factory=dict) # задаёт параметры для поиска (параллелизм, количество итераций)


def resolve_params(name):
    '''Разрешает параметры модели, объединяя сохраненные и заданные вручную. Заданные вручную параметры переопределяют сохраненные.'''
    saved = load_best_params(name) if RUNCONF.use_saved_params else {}
    return {**saved, **MODELS[name].model_params}


def build_model(name):
    """Собирает модель на основе заданной конфигурации"""
    model = MODELS[name].build()
    model.set_params(**resolve_params(name))
    return model


# MODELS ищется только в момент вызова функций выше, поэтому словарь можно объявить после них
MODELS = {
    'ridge': MLModelConf(get_ridge_pipeline, RIDGE_GRID),
    'lasso': MLModelConf(get_lasso_pipeline, LASSO_GRID),
    'elasticnet': MLModelConf(get_elasticnet_pipeline, ELASTICNET_GRID),
    'knn': MLModelConf(get_knn_pipeline, KNN_GRID),
    'tree': MLModelConf(get_tree_pipeline, TREE_GRID),
    'xgb': MLModelConf(get_xgb_pipeline, XGB_GRID,
                       model_params={'xgb__learning_rate': 0.05},
                       search_params={'n_jobs': 1, 'n_iter': 30}),
    'catboost': MLModelConf(get_catboost_pipeline, CATBOOST_GRID,
                            model_params={'catboost__learning_rate': 0.05},
                            search_params={'n_jobs': 1, 'n_iter': 30}),
    'nn': MLModelConf(get_nn_pipeline, NN_GRID,
                      model_params={'nn__hidden_sizes': (100, 50),
                                    'nn__activation': 'leaky_relu',
                                    'nn__optimizer': 'sgd',
                                    'nn__lr': 0.001,
                                    'nn__epochs': 50},
                      search_params={'n_iter': 40}),
    'voting': MLModelConf(partial(get_voting_regressor, build_model, RUNCONF.ensemble_members), search_params={'n_jobs': 1}),
    'stacking': MLModelConf(partial(get_stacking_regressor, build_model, RUNCONF.ensemble_members), search_params={'n_jobs': 1}),
}
