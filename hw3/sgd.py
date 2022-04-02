#################################
# Your name: Daniel Ezer
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
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
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    n = data.shape[0]
    w = np.zeros(data.shape[1])
    for t in range(1, T + 1):
        # Reduce the step size on each iteration
        eta_t = eta_0 / t

        # Uniformly choose a data point
        random_i = np.random.randint(0, n)
        x_i = data[random_i]
        y_i = labels[random_i]

        # If we made a mistake (probably) in the classification, move in the direction of the gradient
        if np.dot(w, x_i * y_i) <= 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i
        # If the classification was correct, scale the norm of w
        else:
            w = (1 - eta_t) * w

    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    # TODO: Implement me
    pass

#################################

# Place for additional code

#################################


def get_prediction_accuracy_for_eta(train_data, train_labels, validation_data, validation_labels, C, eta, T):
    n = 10
    s = 0
    for _ in range(n):
        w = SGD_hinge(train_data, train_labels, C, eta, T)
        # Predict the label with sign(w * x)
        predictions = np.sign(np.dot(w, validation_data.T))
        s += np.average(np.equal(predictions, validation_labels))
    return s / n


def q1a(train_data, train_labels, validation_data, validation_labels):
    etas = np.linspace(start=0.01, stop=1, num=20)
    C, T = 1, 1000

    # Calculate the accuracy for each eta
    vfunc = np.vectorize(lambda eta: get_prediction_accuracy_for_eta(train_data, train_labels,
                                                                     validation_data, validation_labels, C, eta, T))
    accuracy = vfunc(etas)

    # Plot the Accuracy data as a function of eta_0
    plot_with_lims(etas, "eta_0", accuracy, "Accuracy", "Prediction Accuracy as a function of eta_0",
                   (-0.1, 1.1), (0.9, 1))

    # Return the eta that results in the best accuracy
    return etas[np.argmax(accuracy)], np.max(accuracy)


def q1b(train_data, train_labels, validation_data, validation_labels):
    cs = np.logspace(start=-5, stop=0, num=20)
    eta, T = 0.739, 1000
    # Calculate the accuracy for each eta
    vfunc = np.vectorize(lambda c: get_prediction_accuracy_for_eta(train_data, train_labels,
                                                                   validation_data, validation_labels, c, eta, T))
    accuracy = vfunc(cs)

    # Plot the Accuracy data as a function of C
    plot_with_lims(cs, "C", accuracy, "Accuracy", "Prediction Accuracy as a function of C")
    plot_with_lims(cs, "C", accuracy, "Accuracy", "Prediction Accuracy as a function of C", (-0.0001, 0.01))

    # Return the C that results in the best accuracy
    return cs[np.argmax(accuracy)], np.max(accuracy)


def plot_with_lims(xv, x_name, yv, y_name, title, x_lim=None, y_lim=None):
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.scatter(xv, yv)
    plt.plot(xv, yv)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.savefig("q1b.png")
    plt.show()


if __name__ == '__main__':
    # Get training and validation data
    train_d, train_l, validation_d, validation_l, test_d, test_l = helper()

    print(q1b(train_d, train_l, validation_d, validation_l))
