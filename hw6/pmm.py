import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np


REAL_CS = [0.4, 0.4, 0.2]
REAL_LAMBS = [5, 10, 18]


def generate_data_from_pmm(num_samples, cs, lambdas):
    z = np.random.choice(lambdas, size=num_samples, p=cs)
    return np.vectorize(np.random.poisson)(z)


def q2a():
    return generate_data_from_pmm(1000, REAL_CS, REAL_LAMBS)


def r(x: int, lam: int, c: float) -> float:
    return poisson.pmf(x, lam) * c


def p(x, z, lam, c):
    return r(x, lam[z], c[z]) / np.sum(np.vectorize(r, excluded=['x'])(x=x, lam=lam, c=c))


def em(sample, num_iterations):
    n = sample.shape[0]
    k = 3
    c_t = np.ones(k) * (1/k)
    lam_t = np.arange(start=1, stop=k + 1)
    c_t_p_1 = np.zeros(k)
    lam_t_p_1 = np.zeros(k)
    for t in range(num_iterations):
        for z in range(len(lam_t)):
            ps = np.vectorize(p, excluded=['z', 'lam', 'c'])(sample, z=z, lam=lam_t, c=c_t)
            p_sum_over_x = np.sum(ps)
            lam_t_p_1[z] = np.sum(ps * sample) / p_sum_over_x
            c_t_p_1[z] = p_sum_over_x / n
        lam_t = lam_t_p_1
        c_t = c_t_p_1

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
        em_ys = np.vectorize(lambda x: np.sum([cs[i] * poisson.pmf(x, lams[i]) for i in range(len(lams))]))(xs)
        real_ys = np.vectorize(lambda x: np.sum([REAL_CS[i] * poisson.pmf(x, REAL_LAMBS[i])
                                                 for i in range(len(REAL_LAMBS))]))(xs)
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
    print("HI")
