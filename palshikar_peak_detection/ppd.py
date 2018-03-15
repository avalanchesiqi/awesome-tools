# -*- coding: utf-8 -*-
"""
==============================================================================
The code for the class ppd
==============================================================================
This is the main class of Palshikar Peak Detection framework.
It detects peaks in a time series data stream. It works on any scale, not just probability stream bounded within [0, 1].
Here I only examine the second scoring function: compute the average signed distance to +w/-w temporal neighbours.
Paper: Palshikar, G., 2009, June. Simple algorithms for peak detection in time-series.
In Proc. 1st Int. Conf. Advanced Data Analysis, Business Analytics and Intelligence (pp. 1-13).
http://constans.pbworks.com/w/file/fetch/120908295/Simple_Algorithms_for_Peak_Detection_in_Time-Serie.pdf
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# pretty font size
mpl.rcParams.update({'axes.titlesize': 14,
                     'font.size': 12})


class PPD(object):
    """ The main class of Palshikar Peak Detection framework.
    It detects peaks in a time series data stream.
    """
    def __init__(self, data, w, h, score_func):
        """ Initialize class.
        :param data: time series data for detecting peaks
        :param w: nearby w neighbours to compare
        :param h: minimal distance to mean value in the unit of standard deviation
        :param score_func: scoring function to compute the peak intensity for each time step
        """
        self.data = np.array(data)
        self.n = len(self.data)
        self.w = w
        self.h = h
        self.score_func = score_func
        self.peak_scores = None
        self.peaks = None

    def print_peaks(self):
        """ Print peak times.
        """
        print('detected peaks:')
        print('+-----------+----------+')
        print('| peak time | distance |')
        for t in self.peaks:
            print('| {0: >9} | {1: >8} |'.format(t, self.peak_scores[t]))
        print('+-----------+----------+')

    # == == == == == == == == modelling components == == == == == == == == #
    def detect_burst(self):
        """ Find peaks by measuring the distance to mean value of scoring function for each time step.
        """
        self.peak_scores = np.array([self.score_func(self.data, t, self.w) for t in range(self.n)])
        positive_peak_scores = self.peak_scores[self.peak_scores > 0]
        mu = np.mean(positive_peak_scores)
        sigma = np.std(positive_peak_scores)
        # filter peak times
        peak_array = np.array([1 if (self.peak_scores[t] - mu) > self.h * sigma else 0 for t in range(self.n)])
        peak_times = np.argwhere(peak_array == 1).flatten()
        if len(peak_times) > 0:
            ret = [peak_times[0]]
            m = len(peak_times)
            for j in range(1, m):
                if peak_times[j] - ret[-1] <= self.w:
                    if self.data[ret[-1]] < self.data[peak_times[j]]:
                        ret.pop(-1)
                        ret.append(peak_times[j])
                else:
                    ret.append(peak_times[j])
            self.peaks = ret

    # plot function for time series data and peaks detection results
    def plot_func(self):
        """ Plot peak detection results of PPD framework.
        """
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)

        ax1.plot(np.arange(self.n), self.data, 'k--', lw=1, label='time series data')
        ax1.scatter(self.peaks, self.data[self.peaks], marker='x', color='r', zorder=30, label='detected peaks')

        ax1.set_xlabel('video age')
        ax1.set_ylabel('daily view')
        ax1.set_xlim(xmin=0)
        ax1.set_ylim(ymin=0)
        ax1.tick_params(axis='y', rotation=90)
        ax1.set_yticks(ax1.get_yticks()[::3])
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{0:.0f}K'.format(y / 1000)))

        plt.legend(loc='upper left', frameon=False)
        plt.tight_layout()
        plt.show()
