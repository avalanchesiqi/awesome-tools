#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The code for random walk model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf


def random_walker_model():
    num_steps = 1000
    start_point = 5
    random_walker = [start_point]
    for i in range(num_steps):
        random_walker.append(random_walker[-1] + np.random.random() - 0.5)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))

    # observation vs time step
    axes[0].plot(range(1 + num_steps), random_walker)
    axes[0].set_ylabel('observation')

    # diff1 vs time step
    axes[1].plot(range(num_steps), np.diff(random_walker))
    axes[1].set_xlabel('time steps')
    axes[1].set_ylabel('diff1')

    # autocorrelation in diff1
    plot_pacf(np.diff(random_walker), lags=25, ax=axes[2])
    axes[2].set_xlabel('lag')

    plt.show()


def geometric_random_walker_model():
    sp500 = pd.read_csv('./data/sp500_data.csv', sep=',', header=0)
    sp500.set_index(pd.DatetimeIndex(sp500['Date']), inplace=True)
    sp500_index = sp500['SP500']

    fig, axes = plt.subplots(6, 1, figsize=(8, 9))

    # raw data
    axes[0].plot(sp500_index)
    axes[0].set_title('SP500 monthly')

    # diff1
    axes[1].plot(sp500_index.diff().dropna())
    # axes[1].set_xlabel('time steps')
    axes[1].set_title('Diff(SP500 monthly)')

    # acf in diff1
    plot_pacf(sp500_index.diff().dropna(), lags=25, ax=axes[2])
    axes[2].set_title('ACF for Diff(SP500 monthly)')

    # log10 of raw data
    axes[3].plot(sp500_index.apply(np.log10))
    axes[3].set_title('Log(SP500 monthly)')

    # log10 of diff1
    axes[4].plot(sp500_index.apply(np.log10).diff().dropna())
    axes[4].set_title('Diff(Log(SP500 monthly))')

    # acf in log10 diff1
    plot_pacf(sp500_index.apply(np.log10).diff().dropna(), lags=25, ax=axes[5])
    axes[5].set_title('ACF for Diff(Log(SP500 monthly))')

    plt.show()


if __name__ == '__main__':
    random_walker_model()
    geometric_random_walker_model()
