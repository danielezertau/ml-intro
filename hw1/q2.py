import numpy
from sklearn.datasets import fetch_openml
from matplotlib import pyplot


def get_mnist_data():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']
    return data, labels


def get_train_and_test_sets(data, labels, num_train_images, num_test_images=1000):
    num_images = num_train_images + num_test_images
    idx = numpy.random.RandomState(0).choice(70000, num_images)
    train = data[idx[:num_train_images], :].astype(int)
    train_labels = labels[idx[:num_train_images]]
    test = data[idx[num_images - num_test_images:], :].astype(int)
    test_labels = labels[idx[num_images - num_test_images:]]
    return train, train_labels, test, test_labels


def knn(train, train_labels, image, k):
    euclidean_distance = numpy.apply_along_axis(numpy.linalg.norm, 1, train - image)
    k_nearest_indices = numpy.argpartition(euclidean_distance, k)[:k]
    return numpy.bincount(train_labels[k_nearest_indices].astype(int)).argmax()


def accuracy_for_k(k, train, train_labels, test, test_labels):
    results = numpy.apply_along_axis(lambda image: knn(train, train_labels, image, k), 1, test)
    return (numpy.count_nonzero(results == test_labels.astype(int)) / len(test_labels)) * 100


def q2b(train, train_labels, test, test_labels):
    return accuracy_for_k(10, train, train_labels, test, test_labels)


def q2c(train, train_labels, test, test_labels):
    ks = numpy.arange(1, 101)
    results = numpy.array([accuracy_for_k(k, train, train_labels, test, test_labels) for k in ks])
    pyplot.xlabel('K')
    pyplot.ylabel('Predictor accuracy')
    pyplot.plot(ks, results)
    pyplot.savefig('q2c.png')
    pyplot.show()
    return results.argmax(), numpy.max(results)


def q2d(data, labels):
    def accuracy(num_train_images):
        train, train_labels, test, test_labels = get_train_and_test_sets(data, labels, num_train_images)
        return accuracy_for_k(1, train, train_labels, test, test_labels)
    ns = numpy.arange(100, 5001, 100)
    results = numpy.array([accuracy(n) for n in ns])
    pyplot.xlabel('N')
    pyplot.ylabel('Predictor accuracy')
    pyplot.plot(ns, results)
    pyplot.savefig('q2d.png')
    pyplot.show()


if __name__ == '__main__':
    n_train_images = 1000
    images, images_labels = get_mnist_data()
    train_images, train_images_labels, test_images, test_images_labels = get_train_and_test_sets(images, images_labels,
                                                                                                 n_train_images)
    print(q2b(train_images, train_images_labels, test_images, test_images_labels))
    print(q2c(train_images, train_images_labels, test_images, test_images_labels))
    q2d(images, images_labels)
