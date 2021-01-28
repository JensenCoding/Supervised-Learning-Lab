import numpy as np
from sklearn import neighbors
from .Perform import exp_perform, plot_validation_curve


def KNN(data):
    params = {'metric': ['manhattan', 'euclidean', 'chebyshev'], 'n_neighbors': np.arange(1, 21, 1),
              'weights': ['uniform', 'distance']}
    v_param = {'n_neighbors': np.arange(1, 21, 1)}

    learner = neighbors.KNeighborsClassifier()

    best_params = exp_perform(data, learner, 'KNN', params, validation_params=v_param)

    return best_params

