import math
import numpy
from scipy.stats import bernoulli
from matplotlib import pyplot


def generate_bernoulli_matrix(success_prob, num_samples, sample_size):
    return numpy.array([bernoulli.rvs(success_prob, size=sample_size) for _ in range(num_samples)])


def plot_emp_hoeffding(success_prob, num_samples, sample_size):
    bernoulli_matrix = generate_bernoulli_matrix(success_prob, num_samples, sample_size)
    empirical_average = numpy.apply_along_axis(numpy.sum, 1, bernoulli_matrix) / sample_size
    epsilons = numpy.linspace(0, 1, 3000)
    dist_from_exp = numpy.abs(empirical_average - success_prob)
    emp_dist_prob = numpy.array([numpy.count_nonzero(dist_from_exp > eps) for eps in epsilons]) / num_samples
    hoeffding_dist_prob = math.e ** (-2 * sample_size * (epsilons ** 2))
    pyplot.xlabel('epsilon')
    pyplot.ylabel('P(|X_i - 0.5| > epsilon)')
    pyplot.plot(epsilons, emp_dist_prob, label='Empirical results')
    pyplot.plot(epsilons, hoeffding_dist_prob, label='Hoeffding bound')
    pyplot.legend()
    pyplot.savefig('hoeffding.png')
    pyplot.show()


if __name__ == '__main__':
    p = 0.5
    N = 2 * (10 ** 5)
    n = 20
    plot_emp_hoeffding(p, N, n)
