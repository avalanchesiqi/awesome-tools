# -*- coding: utf-8 -*-
"""
==============================================================================
The code for the class spb
==============================================================================
This is the main class of Sleeping Beauties model.
It identifies time of awakening and time of falling for the global peak.
Paper: Ke, Q., Ferrara, E., Radicchi, F. and Flammini, A., 2015.
Defining and identifying Sleeping Beauties in science. In PNAS, 7426-7431
http://www.pnas.org/content/112/24/7426.short
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# pretty font size
mpl.rcParams.update({'axes.titlesize': 14,
                     'font.size': 12})


class SPB(object):
    """ The main class of Sleeping Beauties model.
    It identifies time of awakening and time of falling for the global peak.
    """
    def __init__(self, data):
        """ Initialize class with time series data.
        :param data: time series data for identifying sleep beauties
        """
        self.data = np.array(data)
        # normalize the time series data to the scale of length
        self.scale = max(self.data) / len(self.data)
        self.norm_data = self.data / self.scale
        self.start = 0
        self.end = len(self.norm_data) - 1
        self.peak = np.argmax(self.norm_data)
        self.awakening = None
        self.falling = None
        self.awakening_proj_coordination = None
        self.falling_proj_coordination = None

    # get parameters from model
    def get_parameters(self):
        """ Get parameters from model.
        :return: model parameters of start, t_awakening, t_peak, t_falling, end
        """
        return np.array([self.start, self.awakening, self.peak, self.falling, self.end])

    def print_parameters(self):
        """ Print model parameters.
        """
        print('start={0}, t_awakening={1}, t_peak={2}, t_falling={3}, end={4}'
              .format(*self.get_parameters()))

    # == == == == == == == == modelling components == == == == == == == == #
    @staticmethod
    def dist_and_proj(line_params, point_coordination):
        """ Orthogonal distance |ax_0 + by_0 + c| / sqrt(a^2 + b^2) for line ax+by+c=0 and point (x_0, y_0).
        source: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        :param line_params: parameters for line ax+by+c=0
        :param point_coordination: coordination for point (x_0, y_0)
        :return: Orthogonal distance from point to line and projection point coordination
        """
        a, b, c = line_params
        x_0, y_0 = point_coordination
        dist = abs(a * x_0 + b * y_0 + c) / np.sqrt(a ** 2 + b ** 2)
        projection_coordination = [(-a * b * y_0 + b ** 2 * x_0 - a * c) / (a ** 2 + b ** 2),
                                   (a ** 2 * y_0 - a * b * x_0 - b * c) / (a ** 2 + b ** 2)]
        return dist, projection_coordination

    def identify_awakening_and_falling(self):
        """ Identify awakening time, falling time, awakening projection and falling projection, and set to model.
        """
        # rising phase
        rise_a = (self.norm_data[self.peak] - self.norm_data[self.start]) / (self.peak - self.start)
        rise_b = -1
        rise_c = self.norm_data[self.start] - (self.norm_data[self.peak] - self.norm_data[self.start]) / (self.peak - self.start) * self.start
        rise_refer_line = lambda t: rise_a * t + rise_c
        rise_refer_line_params = [rise_a, rise_b, rise_c]
        t_awakening = np.argmax([self.dist_and_proj(rise_refer_line_params, (x_0, y_0))[0]
                                 for x_0, y_0 in enumerate(self.norm_data[self.start: self.peak])])
        proj_awakening_coordination = self.dist_and_proj(rise_refer_line_params, (t_awakening, self.norm_data[t_awakening]))[1]
        rise_b_coefficient = np.sum([(rise_refer_line(t) - self.norm_data[t]) / max(1, self.norm_data[t]) for t in range(self.start, self.peak)]) * self.scale
        self.awakening = t_awakening
        self.awakening_proj_coordination = proj_awakening_coordination

        # falling phase
        fall_a = (self.norm_data[self.peak] - self.norm_data[self.end]) / (self.peak - self.end)
        fall_b = -1
        fall_c = self.norm_data[self.end] - (self.norm_data[self.peak] - self.norm_data[self.end]) / (self.peak - self.end) * self.end
        fall_refer_line = lambda t: fall_a * t + fall_c
        fall_refer_line_params = [fall_a, fall_b, fall_c]
        t_falling = self.peak + np.argmax([self.dist_and_proj(fall_refer_line_params, (self.peak + x_0, y_0))[0]
                                           for x_0, y_0 in enumerate(self.norm_data[self.peak: self.end + 1])])
        proj_falling_coordination = self.dist_and_proj(fall_refer_line_params, (t_falling, self.norm_data[t_falling]))[1]
        fall_b_coefficient = np.sum([(fall_refer_line(t) - self.norm_data[t]) / max(1, self.norm_data[t]) for t in range(self.peak, self.end + 1)]) * self.scale
        self.falling = t_falling
        self.falling_proj_coordination = proj_falling_coordination

        # print parameters
        self.print_parameters()
        print('rise B_coefficient={0:.2f}, fall B_coefficient={1:.2f}'.format(rise_b_coefficient, fall_b_coefficient))

    # plot function for identifying awakening time, falling time
    def plot_func(self, title):
        """ Plot awakening time and falling time of SPB model.
        :param title: figure title, YoutubeID
        """
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(111)

        ax1.plot(np.arange(len(self.norm_data)), self.norm_data, 'k', label='observed #views')
        ax1.scatter([self.awakening, self.falling], [self.norm_data[self.awakening], self.norm_data[self.falling]],
                    marker='o', color='r', zorder=30)
        ax1.scatter(self.peak, self.norm_data[self.peak], marker='x', color='r', zorder=30)
        ax1.plot([self.start, self.peak], [self.norm_data[self.start], self.norm_data[self.peak]], 'r--')
        ax1.plot([self.end, self.peak], [self.norm_data[self.end], self.norm_data[self.peak]], 'r--')

        ax1.plot([self.awakening, self.awakening_proj_coordination[0]], [self.norm_data[self.awakening], self.awakening_proj_coordination[1]], 'b--')
        ax1.plot([self.falling, self.falling_proj_coordination[0]], [self.norm_data[self.falling], self.falling_proj_coordination[1]], 'b--')
        ax1.text(0.03, 0.80, 't_start={0}\nt_awakening={1}\nt_peak={2}\nt_falling={3}\nt_end={4}'
                 .format(*self.get_parameters()), transform=ax1.transAxes)

        ax1.set_xlabel('days')
        ax1.set_ylabel('normalized views')
        ax1.tick_params(axis='y', rotation=90)
        ax1.set_ylim(ymin=ax1.get_xlim()[0])
        ax1.set_ylim(ymax=ax1.get_xlim()[1])
        ax1.set_title(title)

        plt.tight_layout()
        plt.show()