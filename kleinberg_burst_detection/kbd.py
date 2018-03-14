# -*- coding: utf-8 -*-
"""
==============================================================================
The code for the class kbd
==============================================================================
This is the main class of Kleinberg Burst Detection model.
It identifies periods of different burst states via optimizing latent state given observation.
Sec 3 "Hierarchical Structure and E-mail Streams" introduces setting of continuous events stream.
Here I focus on Sec 4 "Enumerating Bursts", where multiple events happen simultaneously in discrete batch.
For instance, publications in conference every year.
Paper: Kleinberg, J., 2003. Bursty and hierarchical structure in streams. In KDD, 373-397.
https://www.cs.cornell.edu/home/kleinber/bhs.pdf
Blog: https://nikkimarinsek.com/blog/kleinberg-burst-detection-algorithm
"""

from collections import defaultdict
import numpy as np
from scipy.misc import comb
import matplotlib as mpl
import matplotlib.pyplot as plt
# pretty font size
mpl.rcParams.update({'axes.titlesize': 14,
                     'font.size': 12})


class KBD(object):
    """ The main class of Kleinberg Burst Detection model.
    It identifies periods of burst states in discrete batch events.
    """
    def __init__(self):
        """ Initialize class.
        """
        self.d = None
        self.r = None
        self.n = None
        self.k = None
        self.s = None
        self.gamma = None
        self.p = None
        self.q = None
        self.bursts = None

    def initial(self, d, r, k=2):
        """ Initialize or reset model with data.
        :param d: number of events in each time step
        :param r: number of target events in each time step
        :param k: number of state, default 2 states
        """
        if isinstance(k, int) and k >= 2:
            self.d = np.array(d)
            self.r = np.array(r)
            self.n = len(d)
            self.k = k
        else:
            raise Exception('--- Number of states k must be an integer and no less than 2!')

    def set_s(self, s):
        """ Set hyperparameter s.
        :param s: multiplicative distance between probabilities of states
        """
        self.s = s
        self.p = self.get_state_probability()

    def set_gamma(self, gamma):
        """ Set hyperparameter gamma.
        :param gamma: difficulty to move up a state
        """
        self.gamma = gamma

    def burst_weight(self, p_0, p_i, d_t, r_t):
        """ Get burst weight for d_t and r_t in state p_i
        :param p_0: base probability
        :param p_i: target event probability at state i
        :param d_t: number of events in time t
        :param r_t: number of target events in time t
        :return: burst weight
        """
        return self.cost_fit_observation(d_t, r_t, p_0) - self.cost_fit_observation(d_t, r_t, p_i)

    def extract_structure(self):
        """ Extract hierarchical structure of bursts.
        :return: dictionary of burst {state: [[id, start, end, weight]]}
        """
        bursts = defaultdict(list)
        for i in range(1, self.k):
            burst_times = np.argwhere(self.q >= i).flatten()
            bid = 0
            start = burst_times[0]
            last_bt = start
            weight = self.burst_weight(self.p[0], self.p[i], self.d[last_bt], self.r[last_bt])
            for j in range(1, len(burst_times)):
                # print(burst_times[j], last_bt+1)
                if burst_times[j] == last_bt + 1:
                    last_bt = burst_times[j]
                    weight += self.burst_weight(self.p[0], self.p[i], self.d[last_bt], self.r[last_bt])
                else:
                    bursts[i].append([bid, start, last_bt, weight])
                    bid += 1
                    start = burst_times[j]
                    last_bt = start
                    weight = self.burst_weight(self.p[0], self.p[i], self.d[last_bt], self.r[last_bt])
            bursts[i].append([bid, start, last_bt, weight])
        return bursts

    def print_bursts(self):
        """ Print burst structure.
        """
        print('bursty probability:')
        print('+' + '--------+' * self.k)
        print('|' + ''.join([' state{0} |'.format(i) for i in range(self.k)]))
        print('|' + ''.join([' {0:.4f} |'.format(self.p[i]) for i in range(self.k)]))
        print('+' + '--------+' * self.k)

        print('weighted bursts:')
        print('+-------+-------+-------+-------+----------+')
        print('| state | label | begin |   end |  weight  |')
        for i in sorted(self.bursts):
            for burst in self.bursts[i]:
                bid, start, end, weight = burst
                print('| {0: >5} | {1: >5} | {2: >5} | {3: >5} | {4: >#08.4f} |'.format(i, bid, start, end, weight))
        print('+-------+-------+-------+-------+----------+')

    # == == == == == == == == modelling components == == == == == == == == #
    def get_state_probability(self):
        """ State probability of each in self.k states
        :return: list of probability in each state
        """
        if self.r is None:
            raise Exception('--- Initialize data before generating state probability!')
        p_0 = np.sum(self.r) / np.sum(self.d)
        ret = []
        for i in range(self.k):
            p_i = p_0 * self.s ** i
            p_i = p_i if p_i < 1 else 1
            ret.append(p_i)
        return ret

    @staticmethod
    def cost_state_change(i, j, gamma, n):
        """ Cost of changing state, positive if moving up state, otherwise zero.
        :param i: current state
        :param j: next state
        :param gamma: penalty for moving up a state
        :param n: number of timepoints
        :return: cost of changing state
        """
        if i >= j:
            return 0
        else:
            return (j - i) * gamma * np.log(n)

    @staticmethod
    def cost_fit_observation(d_t, r_t, p_i):
        """ Cost of fitting observation, if automation is at state q_i, then t batch incurs cost of this.
        :param d_t: number of events at time t
        :param r_t: number of target events at time t
        :param p_i: probability of in state i
        :return: cost of fitting observation
        """
        return -np.log(comb(d_t, r_t) * (p_i**r_t) * ((1-p_i)**(d_t-r_t)))

    def detect_burst(self):
        """ Find optimal state sequence by Viterbi algorithm.
        """
        # initialize matrices to hold the costs and optimal state sequence
        cost = np.full([self.n, self.k], np.nan)
        q = np.full([self.n, 1], np.nan)

        # use the Viterbi algorithm to find the optimal state sequence
        for t in range(self.n):
            # calculate the cost to transition to each state
            for i in range(self.k):
                # for the first time step, calculate the fit cost only
                if t == 0:
                    cost[t, i] = self.cost_fit_observation(self.d[t], self.r[t], self.p[i])
                # for all other time steps, calculate the fit and transition cost
                else:
                    cost[t, i] = self.cost_state_change(q[t-1], i, self.gamma, self.n) + self.cost_fit_observation(self.d[t], self.r[t], self.p[i])

            # add the state with the minimum cost to the optimal state sequence
            q[t] = np.argmin(cost[t, :])
        self.q = q.flatten()
        self.bursts = self.extract_structure()

    # plot function for time series data and bursty detection results
    def plot_func(self):
        """ Plot bursty detection results of KBD model.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(8, 5), gridspec_kw={'height_ratios': [3, 2]})

        ax1.plot(np.arange(self.n), self.r/self.n, 'k--')
        ax1.set_ylabel('prob target events')

        bursts = self.bursts
        for state in bursts:
            for seg in bursts[state]:
                _, start, end, _ = seg
                ax2.scatter(np.arange(start, end+1), [state]*(end+1-start), marker='x', color='r')
        ax2.set_xlabel('days')
        ax2.set_ylabel('latent state')
        ax2.set_ylim(ymin=0)
        ax2.set_ylim(ymax=self.k)

        plt.tight_layout()
        plt.show()
