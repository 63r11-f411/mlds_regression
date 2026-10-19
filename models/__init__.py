from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from config import RUNCONF
from utility import load_best_params
from .baseline import get_ridge_pipeline, PARAM_GRID as RIDGE_GRID
from .lasso import get_lasso_pipeline, PARAM_GRID as LASSO_GRID
from .elasticnet import get_elasticnet_pipeline, PARAM_GRID as ELASTICNET_GRID

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
    'elasticnet': MLModelConf(get_elasticnet_pipeline, ELASTICNET_GRID)
}
