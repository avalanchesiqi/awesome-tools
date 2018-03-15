#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, json, time
import numpy as np
from datetime import timedelta

from ppd import PPD


def score_func(pivot, neighbours):
    neighbours = np.array(neighbours)
    return np.sum(pivot - neighbours) / 2 / len(neighbours)


def compute_peak_score(ts_data, idx, w):
    if idx < w:
        neighbours = ts_data[: idx + w + 1]
    elif idx > len(ts_data) - w:
        neighbours = ts_data[idx - w:]
    else:
        neighbours = ts_data[idx - w: idx + w + 1]
    return score_func(ts_data[idx], neighbours)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Load data == == == == == == == == #
    data_prefix = './data'
    filename = 'sleep_beauties.json'
    with open(os.path.join(data_prefix, filename), 'r') as fin:
        video = json.loads(fin.readline().rstrip())
        vid = video['id']
        days = video['insights']['days']
        views = video['insights']['dailyView']
        # paddle zero in missing days
        daily_view = [0] * (days[-1] + 1)
        for i in range(len(days)):
            daily_view[days[i]] = views[i]

    # == == == == == == == == Part 2: Detect peaks == == == == == == == == #
    start_time = time.time()

    print('>>> Detect peaks for video: {0}'.format(vid))

    ppd_model = PPD(data=daily_view, w=3, h=3, score_func=compute_peak_score)
    ppd_model.detect_burst()
    print('>>> Total running time: {0}'.format(str(timedelta(seconds=time.time() - start_time)))[:-3])
    ppd_model.plot_func()
