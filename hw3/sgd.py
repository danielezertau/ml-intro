#################################
# Your name: Daniel Ezer
#################################


import numpy as np
import numpy.random
import scipy.special as sc
from sklearn.datasets import fetch_openml
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sp.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sp.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sp.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    return SGD(data, labels, C, eta_0, T, algo="hinge")


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    return SGD(data, labels, 0, eta_0, T, algo="log")
#################################

# Place for additional code

#################################


def hinge_loss_update_rule(w, x_i, y_i, eta_t, C):
    # If we made a mistake (probably) in the classification, move in the direction of the gradient
    if np.dot(w, x_i * y_i) <= 1:
        w = (1 - eta_t) * w + eta_t * C * y_i * x_i
    # If the classification was correct, scale the norm of w
    else:
        w = (1 - eta_t) * w
    return w


def log_loss_update_rule(w, x_i, y_i, eta_t):
    w_xi_yi_dot_product = np.dot(w, x_i * y_i)
    # If we made a mistake in the classification, move in the direction of the gradient
    if w_xi_yi_dot_product <= 0:
        w = w + eta_t * (y_i * x_i) * sc.expit(0 - w_xi_yi_dot_product)
    # If the classification was correct, do nothing
    return w


def SGD(data, labels, C, eta_0, T, algo):
    n = data.shape[0]
    w = np.zeros(data.shape[1])
    for t in range(1, T + 1):
        # Reduce the step size on each iteration
        eta_t = eta_0 / t

        # Uniformly choose a data point
        random_i = np.random.randint(0, n)
        x_i = data[random_i]
        y_i = labels[random_i]

        if algo == "hinge":
            w = hinge_loss_update_rule(w, x_i, y_i, eta_t, C)
        elif algo == "log":
            w = log_loss_update_rule(w, x_i, y_i, eta_t)
        else:
            raise Exception("Wrong algorithm type. Available options are 'hinge' and 'log")

    return w


def get_prediction_accuracy(train_data, train_labels, validation_data, validation_labels, C, eta, T, algo):
    n = 10
    s = 0
    for _ in range(n):
        w = SGD(train_data, train_labels, C, eta, T, algo)
        # Predict the label with sign(w * x)
        predictions = np.sign(np.dot(w, validation_data.T))
        s += np.average(np.equal(predictions, validation_labels))
    return s / n


def find_best_eta_0(train_data, train_labels, validation_data, validation_labels, etas, C, T, algo):
    # Calculate the accuracy for each eta
    vfunc = np.vectorize(lambda eta: get_prediction_accuracy(train_data, train_labels,
                                                             validation_data, validation_labels, C, eta, T, algo))
    accuracy = vfunc(etas)

    # Plot the Accuracy data as a function of eta_0
    plot_with_lims(etas, "eta_0", accuracy, "Accuracy", "Prediction Accuracy as a function of eta_0")
    plot_with_lims(etas, "eta_0", accuracy, "Accuracy", "Prediction Accuracy as a function of eta_0", (-10, 1000))
    plot_with_lims(etas, "eta_0", accuracy, "Accuracy", "Prediction Accuracy as a function of eta_0", (-1, 100))
    plot_with_lims(etas, "eta_0", accuracy, "Accuracy", "Prediction Accuracy as a function of eta_0", (-0.1, 10))
    plot_with_lims(etas, "eta_0", accuracy, "Accuracy", "Prediction Accuracy as a function of eta_0", (-0.01, 1))
    plot_with_lims(etas, "eta_0", accuracy, "Accuracy", "Prediction Accuracy as a function of eta_0", (-0.001, 0.1))

    # Return the eta that results in the best accuracy
    return etas[np.argmax(accuracy)], np.max(accuracy)


def q1a(train_data, train_labels, validation_data, validation_labels):
    etas = np.linspace(start=0.01, stop=1, num=20)
    C, T = 1, 1000

    return find_best_eta_0(train_data, train_labels, validation_data, validation_labels, etas, C, T, algo="hinge")


def q1b(train_data, train_labels, validation_data, validation_labels):
    cs = np.logspace(start=-5, stop=0, num=20)
    eta, T = 0.739, 1000
    # Calculate the accuracy for each eta
    vfunc = np.vectorize(lambda c: get_prediction_accuracy(train_data, train_labels,
                                                           validation_data, validation_labels, c, eta, T, algo="hinge"))
    accuracy = vfunc(cs)

    # Plot the Accuracy data as a function of C
    plot_with_lims(cs, "C", accuracy, "Accuracy", "Prediction Accuracy as a function of C")
    plot_with_lims(cs, "C", accuracy, "Accuracy", "Prediction Accuracy as a function of C", (-0.0001, 0.01))

    # Return the C that results in the best accuracy
    return cs[np.argmax(accuracy)], np.max(accuracy)


def q1c(train_data, train_labels):
    C, eta, T = 0.00011, 0.739, 20000
    w = SGD_hinge(train_data, train_labels, C, eta, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation="nearest")
    plt.savefig("q1c.png")
    plt.show()


def q1d(train_data, train_labels, test_data, test_labels):
    C, eta, T = 0.00011, 0.739, test_data.shape[0]
    return get_prediction_accuracy(train_data, train_labels, test_data, test_labels, C, eta, T, algo="hinge")


def plot_with_lims(xv, x_name, yv, y_name, title, x_lim=None, y_lim=None):
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.scatter(xv, yv)
    plt.plot(xv, yv)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.show()


def q2a(train_data, train_labels, validation_data, validation_labels):
    etas = np.logspace(start=-5, stop=5, num=11)
    T = 1000

    return find_best_eta_0(train_data, train_labels, validation_data, validation_labels, etas, 0, T, algo="log")


def q2b(train_data, train_labels, test_data, test_labels):
    eta, T = 0.01, 20000
    w = SGD_log(train_data, train_labels, eta, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation="nearest")
    plt.savefig("q2b.png")
    plt.show()
    return get_prediction_accuracy(train_data, train_labels, test_data, test_labels, 0, eta, test_data.shape[0],
                                   algo="log")


if __name__ == '__main__':
    # Get training and validation data
    train_d, train_l, validation_d, validation_l, test_d, test_l = helper()

    print(q2b(train_d, train_l, test_d, test_l))
