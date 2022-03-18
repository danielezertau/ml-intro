#################################
# Your name: Daniel Ezer
#################################
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two-dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        xs = np.sort(np.random.uniform(0, 1, m))
        vfunc = np.vectorize(self.y_given_x_sample)
        ys = vfunc(xs)
        return np.vstack((xs, ys)).T

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two-dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        ns = np.arange(m_first, m_last + 1, step)
        results = np.zeros((2, len(ns)))

        for _ in range(T):
            vfunc = np.vectorize(self.single_experiment_erm_given_n)
            results += vfunc(ns, k)

        results = results / T

        true_error = results[0].T
        empirical_error = results[1].T

        self.plot_errors("q1b", ns, "n", true_error, empirical_error)
        return np.vstack((true_error, empirical_error)).T

    def single_experiment_erm_given_n(self, n, k):
        sample = self.sample_from_D(n)
        hypo, best_error = intervals.find_best_interval(sample.T[0], sample.T[1], k)
        return self.calculate_true_error(hypo), best_error / n

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        ks = np.arange(k_first, k_last + 1, step)
        vfunc = np.vectorize(self.single_experiment_erm_given_n)
        results = vfunc(m, ks)

        true_error = results[0].T
        empirical_error = results[1].T

        self.plot_errors("q1c", ks, "k", true_error, empirical_error)
        return (np.argmin(empirical_error) + 1) * step

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # TODO: Implement the loop
        pass

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods
    #################################

    @staticmethod
    def plot_errors(question, xs, xs_name, true_error, empirical_error):
        plt.xlabel(xs_name)
        plt.ylabel("Error Rate")
        plt.title(question.upper())
        plt.plot(xs, true_error, label='True Error')
        plt.plot(xs, empirical_error, label='Empirical Error')
        plt.legend()
        plt.savefig("%s.png".format(question.lower()))
        plt.show()

    @staticmethod
    def calculate_empirical_error(sample, hypothesis):
        error_count = 0
        for x, y in sample:
            for l, u in hypothesis:
                if (l < x < u) and (y == 0):
                    error_count += 1
                    break
            if y == 1:
                error_count += 1
        return error_count / len(sample)

    @staticmethod
    def y_given_x_sample(x):
        if (0 < x < 0.2) or (0.4 < x < 0.6) or (0.8 < x < 1):
            return bernoulli.rvs(0.8)
        else:
            return bernoulli.rvs(0.1)

    @staticmethod
    def get_intervals_complement(intervals_list):
        intervals_list_complement = []
        if intervals_list[0][0] != 0:
            intervals_list_complement.append((0, intervals_list[0][0]))

        for i in range(len(intervals_list) - 1):
            intervals_list_complement.append((intervals_list[i][1], intervals_list[i+1][0]))

        if intervals_list[-1][1] != 1:
            intervals_list_complement.append((intervals_list[-1][1], 1))

        return intervals_list_complement

    @staticmethod
    def get_interval_intersection_size(interval1, interval2):
        # The intervals are disjoint
        if (interval1[1] <= interval2[0]) or (interval2[1] <= interval1[0]):
            return 0
        # One interval is contained in the other
        elif ((interval1[0] <= interval2[0]) and (interval1[1] >= interval2[1])) or \
                ((interval2[0] <= interval1[0]) and (interval2[1] >= interval1[1])):
            return min(interval1[1] - interval1[0], interval2[1] - interval2[0])
        # The intervals partially intersect
        elif ((interval1[0] <= interval2[0]) and (interval1[1] <= interval2[1])) or \
                ((interval2[0] <= interval1[0]) and (interval2[1] <= interval1[1])):
            return min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])
        else:
            raise Exception("Unexpected intervals, they don't intersect nor are disjoint")

    def calc_error_by_label(self, intervals_list, label):
        ep_h = 0
        for interval in intervals_list:
            for i, lu in enumerate([(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]):
                if lu[1] < interval[0]:
                    continue
                if lu[0] > interval[1]:
                    break
                ep_h += self.get_interval_intersection_size((lu[0], lu[1]), interval) * self.error_cost(i % 2, label)
        return ep_h

    def calculate_true_error(self, intervals_list):
        return self.calc_error_by_label(intervals_list, 1) + \
               self.calc_error_by_label(self.get_intervals_complement(intervals_list), 0)

    @staticmethod
    def error_cost(intervals_section, label):
        # intervals_section 0 represents the intervals ([0, 0.2], [0.4, 0.6], [0.8, 1]) and 1 represents the rest
        penalty_list = [
            [0.8, 0.2],
            [0.1, 0.9],
        ]
        return penalty_list[intervals_section][label]


if __name__ == '__main__':
    ass = Assignment2()
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    print(ass.experiment_k_range_erm(1500, 1, 10, 1))
    # ass.experiment_k_range_srm(1500, 1, 10, 1)
    # ass.cross_validation(1500)
