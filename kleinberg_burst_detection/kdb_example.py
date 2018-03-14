#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from datetime import timedelta

from kbd import KBD

if __name__ == '__main__':
    # == == == == == == == == Part 1: Load data == == == == == == == == #
    # simulate the baseline data
    timepoints = 200
    d = pd.Series(np.floor(np.ones(timepoints) * 200 + np.random.normal(scale=40, size=timepoints)))  # total events
    r = pd.Series(np.floor(np.ones(timepoints) * 20 + np.random.normal(scale=10, size=timepoints)))  # target events
    r[r < 0] = 0  # set negative values to 0

    # add some bursts
    r[20:40] = r[20:40] + 60
    r[70:80] = r[70:80] + 100

    # == == == == == == == == Part 2: Identify bursty and hierarchical structure == == == == == == == == #
    start_time = time.time()

    print('>>> Identify bursty and hierarchical structure in synthetic data stream')

    kdb_model = KBD()
    kdb_model.initial(d, r, k=4)
    kdb_model.set_s(s=1.5)
    kdb_model.set_gamma(gamma=2)
    kdb_model.detect_burst()
    kdb_model.print_bursts()
    kdb_model.plot_func()
    print('>>> Total running time: {0}'.format(str(timedelta(seconds=time.time() - start_time)))[:-3])
