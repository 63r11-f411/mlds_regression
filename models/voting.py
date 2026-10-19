from sklearn.ensemble import VotingClassifier


def build_voting_classifier(build_member, member_names, voting='soft', weights=None):
    estimators = [(name, build_member(name)) for name in member_names]
    return VotingClassifier(estimators=estimators, voting=voting, weights=weights)
