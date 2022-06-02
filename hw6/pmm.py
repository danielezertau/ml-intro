from typing import List
import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np


REAL_CS = np.array([0.4, 0.4, 0.2])
REAL_LAMBS = np.array([5, 10, 18])


def generate_data_from_pmm(num_samples, cs, lambdas):
    z = np.random.choice(lambdas, size=num_samples, p=cs)
    return np.vectorize(np.random.poisson)(z)


def q2a():
    return generate_data_from_pmm(1000, REAL_CS, REAL_LAMBS)


def p(x: int, lam: List[int], c: List[float]) -> List[float]:
    return poisson.pmf(x, lam) * c / np.sum(poisson.pmf(x, lam) * c)


def em(sample, num_iterations):
    n = sample.shape[0]
    k = 3
    c_t = np.ones(k) * (1/k)
    lam_t = np.arange(start=1, stop=k + 1)
    for t in range(num_iterations):
        ps = np.vectorize(p, excluded=['lam', 'c'], signature='()->(n)')(sample, lam=lam_t, c=c_t)
        p_sum_over_x = np.sum(ps, axis=0)
        lam_t = np.sum(ps.T * sample, axis=1) / p_sum_over_x
        c_t = p_sum_over_x / n

    return np.array(lam_t), np.array(c_t)


def q2b():
    num_iterations = 1000
    sample = q2a()
    return em(sample, num_iterations)


def q2c():
    sample = q2a()
    ts = np.array([5, 50, 100, 500, 1000])
    xs = np.arange(50)
    for t in ts:
        lams, cs = em(sample, t)
        em_ys = np.vectorize(lambda x: np.sum(cs * poisson.pmf(x, lams)))(xs)
        real_ys = np.vectorize(lambda x: np.sum(REAL_CS * poisson.pmf(x, REAL_LAMBS)))(xs)
        plt.title(f"t = {t}")
        plt.xlabel("x")
        plt.ylabel("P(X=x)")
        plt.plot(xs, em_ys, label="EM PMF")
        plt.plot(xs, real_ys, label="Real PMM PMF")
        plt.legend()
        plt.savefig(f"q2c-t{t}.pdf")
        plt.show()


if __name__ == '__main__':
    q2c()
