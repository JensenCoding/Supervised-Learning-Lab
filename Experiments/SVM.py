import numpy as np
from sklearn import svm
from .Perform import exp_perform, plot_validation_curve


def SVM_RBF(data):
    X, _ = data.get_data()
    num_sample = X.shape[0]

    gamma_fracs = np.logspace(-6, -1, 5)
    tols = np.arange(1e-8, 0.1, 0.01)
    c_values = np.arange(0.001, 1.5, 0.25)
    iters = [int((1e6 / num_sample) / .8) + 1]

    # RBF SVM
    params = {'max_iter': iters, 'tol': tols, 'class_weight': ['balanced'],
              'C': c_values,
              'decision_function_shape': ['ovo', 'ovr'], 'gamma': gamma_fracs}
    v_param = {'gamma': np.logspace(-6, 1, 7)}
    lc_params = {'max_iter': [2**x for x in range(12)]}

    learner = svm.SVC(kernel='rbf')

    best_params = exp_perform(data, learner, 'SVM_RBF', params, validation_params=v_param, lc_params=lc_params)

    return best_params


def SVM_linear(data):
    X, _ = data.get_data()
    num_sample = X.shape[0]

    tols = np.arange(1e-8, 0.1, 0.01)
    c_values = np.arange(0.001, 1.5, 0.25)
    iters = [int((1e6 / num_sample) / .8) + 1]

    # Linear SVM
    params = {'max_iter': iters, 'tol': tols, 'class_weight': ['balanced'],
              'C': c_values}

    v_param = {'C': np.arange(0.1, 2.5, 0.1)}

    lc_params = {'max_iter': [2**x for x in range(12)]}

    learner = svm.LinearSVC(dual=True)

    best_params = exp_perform(data, learner, 'SVM_Linear', params, validation_params=v_param, lc_params=lc_params)

    return best_params
