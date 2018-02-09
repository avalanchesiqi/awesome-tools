"""
How to Check if Time Series Data is Stationary with Python
source: https://machinelearningmastery.com/time-series-data-stationary-python/
"""

from pandas import Series
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # == == == == == == Part 1: prepare exemplary data == == == == == == #
    # stationary time series data
    s_data = Series.from_csv('./data/stationary_ts/daily-total-female-births.csv', header=0)
    # non-stationary time series data
    ns_data = Series.from_csv('./data/stationary_ts/international-airline-passengers.csv', header=0)

    # visualize data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    ax1.plot(s_data)
    ax1.set_title('Stationary time series data')
    ax2.plot(ns_data)
    ax2.set_title('Non-stationary time series data')
    plt.tight_layout()
    plt.show()

    # == == == == == == Part 2: calculate summary statistics == == == == == == #
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    ax11 = ax1.twinx()
    ax22 = ax2.twinx()
    # split the time series into two continuous sequences and calculate mean and variance
    for idx, data in enumerate((s_data, ns_data)):
        X = data.values
        split = len(X) // 2
        X1, X2 = X[:split], X[split:]
        if idx == 0:
            print('statistics for two continuous sequences in stationary data')
            ax11.hist(X1, normed=True, cumulative=True, histtype='step', color='b', label='CDF of 1st half')
            ax11.hist(X2, normed=True, cumulative=True, histtype='step', color='r', label='CDF of 2nd half')
        else:
            print('statistics for two continuous sequences in non-stationary data')
            ax22.hist(X1, normed=True, cumulative=True, histtype='step', color='b', label='CDF of 1st half')
            ax22.hist(X2, normed=True, cumulative=True, histtype='step', color='r', label='CDF of 2nd half')
        mean1, mean2 = X1.mean(), X2.mean()
        var1, var2 = X1.var(), X2.var()
        print('mean1=%.2f, mean2=%.2f' % (mean1, mean2))
        print('variance1=%.2f, variance2=%.2f' % (var1, var2))
        print('running Kolmogorov-Smirnov statistic test...')
        print(ks_2samp(X1, X2))
        # plot CDF of two sequences
        print('-'*79)

    ax1.hist(s_data, alpha=0.2, color='k')
    ax1.set_title('Stationary data ~ Gaussian distribution')
    ax2.hist(ns_data, alpha=0.2, color='k')
    ax2.set_title('Non-stationary data ~ Non-Gaussian distribution')
    ax11.legend(loc='best', frameon=False)
    ax22.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()

    # == == == == == == Part 3: perform Augmented Dickey-Fuller test == == == == == == #
    # Null Hypothesis (H0): If accepted (p-value > 0.05), it suggests the time series has a unit root,
    # meaning it is non-stationary. It has some time dependent structure.
    # Alternate Hypothesis (H1): If rejected (p-value <= 0.05), it suggests the time series does not have a unit root,
    # meaning it is stationary. It does not have time-dependent structure.
    for idx, data in enumerate((s_data, ns_data)):
        if idx == 0:
            print('Augmented Dickey-Fuller test for stationary data')
        else:
            print('Augmented Dickey-Fuller test for non-stationary data')
        result = adfuller(data)
        print('ADF Statistic: %.4f' % result[0])
        print('p-value: %.4f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.4f' % (key, value))
        if result[1] > 0.05:
            print('>>> accept H0, it is non-stationary time series data.')
        else:
            print('>>> reject H0, it is stationary time series data.')
        print('='*79)
