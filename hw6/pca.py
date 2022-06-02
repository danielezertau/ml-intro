import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people


def plot_vector_as_image(images, h, w, rows, cols, titles):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimensions of original pi
    """
    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    i = 0
    for row in ax:
        for col in row:
            col.set_xticks([])
            col.set_yticks([])
            col.set_title(titles[i], size=10)
            col.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            i += 1
    plt.suptitle(titles[-1])
    plt.tight_layout()
    plt.savefig(f'{titles[-1].replace(" ", "")}.pdf')
    plt.show()


def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if target == target_label:
            image_vector = image.flatten()
            selected_images.append(image_vector)
    return selected_images, h, w


def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people


######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""


def get_cov_matrix(x):
    mu = np.mean(x, axis=0)
    new_x = x - mu
    return new_x.T @ new_x


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimensions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimension of the matrix
        would be (10,100).
        k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest
       k eigenvalues of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    sigma = get_cov_matrix(X)
    U, S, _ = np.linalg.svd(sigma)
    return U.T[:k, :], S[:k]


def q1b(name):
    X, h, w = get_pictures_by_name(name=name)
    U, _ = PCA(np.array(X), 10)
    plot_vector_as_image(U, h, w, 2, 5, [f"Eigen Vector {i}" for i in range(1, 11)] + [f"Name: {name}"])


def q1c(name):
    X, h, w = get_pictures_by_name(name=name)
    X = np.array(X)
    ks = np.array([1, 5, 10, 30, 50, 100])
    rand_idx = np.random.choice(X.shape[0], size=5, replace=False)
    rand_images = X[rand_idx]
    l2_dists = []
    for k in ks:
        U, _ = PCA(X, k)
        transformed_images = np.apply_along_axis(lambda image: U.T @ U @ image, 1, rand_images)
        l2_dists.append(np.sum(np.apply_along_axis(np.linalg.norm, 1, transformed_images - rand_images)))
        plot_vector_as_image(np.vstack((rand_images, transformed_images)), h, w, 2, 5,
                             ["Before"] * 5 + ["After"] * 5 + [f"Name: {name}, K: {k}"])

    plt.xlabel("K")
    plt.ylabel("L2 Distance Sum")
    plt.scatter(ks, l2_dists)
    plt.plot(ks, l2_dists)
    plt.title("L2 Distance")
    plt.savefig("q1c-l2.pdf")
    plt.show()


if __name__ == '__main__':
    name = "George W Bush"
    q1b(name)
    q1c(name)
