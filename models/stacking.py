from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from utility import make_cv


def get_stacking_regressor(build_member, member_names):
    estimators = [(name, build_member(name)) for name in member_names]
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=make_cv())
