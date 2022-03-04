import numpy
from sklearn.datasets import fetch_openml
from matplotlib import pyplot


def pre_processing(num_train_images):
    num_images = 11000
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']
    idx = numpy.random.RandomState(0).choice(70000, num_images)
    train = data[idx[:num_train_images], :].astype(int)
    train_labels = labels[idx[:num_train_images]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]
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


if __name__ == '__main__':
    num_train_images1 = 1000
    train1, train_labels1, test1, test_labels1 = pre_processing(num_train_images1)
    print(q2c(train1, train_labels1, test1, test_labels1))
