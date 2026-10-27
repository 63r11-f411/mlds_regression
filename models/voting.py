from sklearn.ensemble import VotingRegressor


def get_voting_regressor(build_member, member_names, weights=None):
    estimators = [(name, build_member(name)) for name in member_names]
    return VotingRegressor(estimators=estimators, weights=weights)
