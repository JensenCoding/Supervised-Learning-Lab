import numpy as np
from sklearn import neural_network
from .Perform import exp_perform, plot_validation_curve


def ANN(data):
    d = data.X.shape[1]

    alphas = [10 ** -x for x in np.arange(-1, 6.01, 0.5)]
    hidden_layers = [(h,) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]
    learning_rates = sorted([(2 ** x) / 1000 for x in range(8)] + [0.000001])

    params = {'activation': ['relu', 'logistic'], 'alpha': alphas,
              'learning_rate_init': learning_rates,
              'hidden_layer_sizes': hidden_layers}

    timing_params = {'early_stopping': False}

    v_params = {'alpha': alphas}

    lc_params = {'max_iter': [2**x for x in range(12)]}

    # cross validation
    learner = neural_network.MLPClassifier(max_iter=3000, early_stopping=True, random_state=data.seed)
    cv_best_params = exp_perform(data, learner, 'ANN', params, timing_params=timing_params, validation_params=v_params, lc_params=lc_params)

    return cv_best_params
