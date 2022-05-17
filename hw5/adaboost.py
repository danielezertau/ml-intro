#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns:

        hypotheses :
            A list of T tuples describing the hypotheses chosen by the algorithm.
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals :
            A list of T float values, which are the alpha values obtained in every
            iteration of the algorithm.
    """
    d_t = np.ones(X_train.shape[0]) * (1 / X_train.shape[0])
    w_ts = np.zeros(T)
    h_ts = np.empty(T, dtype=object)
    for t in range(T):
        err_t, h_t = get_weak_learner(d_t, X_train, y_train)
        w_t = 0.5 * np.log((1 - err_t) / err_t)
        h_ts[t] = h_t
        w_ts[t] = w_t
        predictions = np.apply_along_axis(h_t, 1, X_train)
        exp_power = -1 * w_t * y_train * predictions
        d_t = (d_t * np.exp(exp_power)) / np.dot(d_t.T, np.exp(exp_power))
    return h_ts, w_ts


def plot_errors(h_ts, w_ts, X_train, y_train, X_test, y_test):
    train_err = np.zeros(h_ts.shape[0])
    test_err = np.zeros(h_ts.shape[0])
    ts = np.arange(h_ts.shape[0])
    for t in ts:
        h = lambda x: predict_linear_comb(h_ts[:t + 1], w_ts[:t + 1], x)
        train_predictions = np.apply_along_axis(h, 1, X_train)
        train_err[t] = np.sum(train_predictions != y_train) / y_train.shape[0]
        test_predictions = np.apply_along_axis(h, 1, X_test)
        test_err[t] = np.sum(test_predictions != y_test) / y_test.shape[0]

    plt.xlabel('t')
    plt.ylabel('Error Rate')
    plt.plot(ts, train_err, label='Training Error')
    plt.plot(ts, test_err, label='Test Error')
    plt.legend()
    plt.savefig('q1a.pdf')
    plt.show()


def predict_linear_comb(hypotheses, weights, x):
    predictions = np.zeros(hypotheses.shape[0])
    for i, h in enumerate(hypotheses):
        predictions[i] = h(x)
    return 1 if np.dot(predictions.T, weights) >= 0 else -1


def h_predict(x, theta):
    return np.where(x <= theta, 1, -1)


def get_emp_error(h_prediction, y_train, d_t):
    return np.dot(h_prediction.T != y_train, d_t)


def get_weak_learner_w_j(d_t, X_train, y_train, j):
    w_j_cnts = X_train[:, j]
    w_j_unique_cnts = np.unique(w_j_cnts)

    vfunc = np.vectorize(h_predict, excluded=['x'], signature='()->(n)')
    predictions = vfunc(x=w_j_cnts, theta=w_j_unique_cnts)
    emp_errors = np.apply_along_axis(get_emp_error, 1, predictions, y_train=y_train, d_t=d_t)
    emp_errors_cmp = 1 - emp_errors
    min_error = min(np.min(emp_errors), np.min(emp_errors_cmp))
    if np.min(emp_errors) <= np.min(emp_errors_cmp):
        return np.array([min_error, lambda x: 1 if x[j] <= w_j_cnts[np.argmin(emp_errors)] else -1])
    else:
        return np.array([min_error, lambda x: 1 if x[j] > w_j_cnts[np.argmin(emp_errors_cmp)] else -1])


def get_weak_learner(d_t, X_train, y_train):
    js = np.arange(0, 5000)
    vfunc = np.vectorize(get_weak_learner_w_j, excluded=['d_t', 'X_train', 'y_train'], signature='()->(2)')
    hypotheses = vfunc(d_t=d_t, X_train=X_train, y_train=y_train, j=js)
    return hypotheses[np.argmin(hypotheses[:, 0])]


def q1a(X_train, y_train, X_test, y_test):
    T = 80
    hypotheses, weights = run_adaboost(X_train, y_train, T)
    plot_errors(hypotheses, weights, X_train, y_train, X_test, y_test)


def q1b(X_train, y_train):
    T = 10
    hypotheses, weights = run_adaboost(X_train, y_train, T)


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    q1a(X_train, y_train, X_test, y_test)

    ##############################################
    # You can add more methods here, if needed.
    ##############################################


if __name__ == '__main__':
    main()
