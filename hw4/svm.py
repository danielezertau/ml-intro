import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.savefig("q1b.pdf")
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_kernels(sample_data, sample_labels, c_reg_param, coef0):
    linear_kernel = svm.SVC(C=c_reg_param, kernel="linear")
    poly2_kernel = svm.SVC(C=c_reg_param, kernel="poly", degree=2, coef0=coef0)
    poly3_kernel = svm.SVC(C=c_reg_param, kernel="poly", degree=3, coef0=coef0)
    fitted_estimators = np.array([
        linear_kernel.fit(sample_data, sample_labels), poly2_kernel.fit(sample_data, sample_labels),
        poly3_kernel.fit(sample_data, sample_labels)])
    model_names = np.array(["linear", "poly2", "poly3"])
    plot_results(fitted_estimators, model_names, sample_data, sample_labels)


def q1a(sample_data, sample_labels, c_reg_param):
    plot_kernels(sample_data, sample_labels, c_reg_param, 0)


def q1b(sample_data, sample_labels, c_reg_param):
    plot_kernels(sample_data, sample_labels, c_reg_param, 1)


def generate_train_data(num_samples):
    # Data is labeled by a circle
    radius = np.hstack([np.random.random(num_samples), np.random.random(num_samples) + 1.5])
    angles = 2 * math.pi * np.random.random(2 * num_samples)
    x1 = (radius * np.cos(angles)).reshape((2 * num_samples, 1))
    x2 = (radius * np.sin(angles)).reshape((2 * num_samples, 1))

    x = np.concatenate([x1, x2], axis=1)
    y = np.concatenate([np.ones((num_samples, 1)), -np.ones((num_samples, 1))], axis=0).reshape([-1])
    return x, y


if __name__ == '__main__':
    C_hard = 1000000.0  # SVM regularization parameter
    C = 10
    n = 100
    x, y = generate_train_data(n)
    q1b(x, y, C)
