#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, json, time
from datetime import timedelta

from hip import HIP

if __name__ == '__main__':
    # == == == == == == == == Part 1: Load data == == == == == == == == #
    data_prefix = './data'
    filename = 'popular_viral.json'
    with open(os.path.join(data_prefix, filename), 'r') as fin:
        video = json.loads(fin.readline().rstrip())
        vid = video['id']
        daily_share = video['insights']['dailyShare']
        daily_view = video['insights']['dailyView']
        daily_watch = video['insights']['dailyWatch']

    # == == == == == == == == Part 2: Fit model and forecast future volume == == == == == == == == #
    start_time = time.time()

    print('>>> Fitting and forecasting for video: {0}'.format(vid))

    # uncomment next 2 lines if fit daily watch time
    # # convert watch time to hour unit
    # daily_watch = [x / 60 for x in daily_watch]

    num_train = 90
    num_test = 30
    num_initialization = 25

    hip_model = HIP()
    hip_model.initial(daily_share, daily_view, num_train, num_test, num_initialization)
    hip_model.fit_with_bfgs()
    hip_model.print_parameters()
    print('>>> Total fitting time: {0}'.format(str(timedelta(seconds=time.time() - start_time)))[:-3])
    hip_model.plot_func('YouTubeID={0}'.format(vid))
