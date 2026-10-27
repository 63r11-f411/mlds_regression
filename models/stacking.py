from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from utility import make_cv

META_MODELS = {'linear': LinearRegression, 'ridge': Ridge}


def get_stacking_regressor(build_member, member_names, meta_model='ridge'):
    '''
    Создает пайплайн для стекинга регрессоров с указанной мета-моделью
    '''
    estimators = [(name, build_member(name)) for name in member_names]
    return StackingRegressor(estimators=estimators, final_estimator=META_MODELS[meta_model](), cv=make_cv())