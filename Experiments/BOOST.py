import numpy as np
from sklearn import ensemble, tree
from .Perform import exp_perform, plot_validation_curve


def BOOST(data):
    seed = data.seed
    # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/Boosting.py
    dt = tree.DecisionTreeClassifier(class_weight='balanced', random_state=seed)
    booster = ensemble.AdaBoostClassifier(algorithm='SAMME', learning_rate=1, base_estimator=dt, random_state=seed)

    params = {'n_estimators': [1, 2, 5, 10, 20, 30, 45, 60, 80, 90, 100],
              'learning_rate': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1],
              'base_estimator__max_depth': np.arange(1, 11, 1),
              'base_estimator__criterion': ['gini', 'entropy']}

    v_params = {'learning_rate': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]}

    lc_params = {'n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

    best_params = exp_perform(data, booster, 'Boost', params, validation_params=v_params, lc_params=lc_params)

    return best_params
