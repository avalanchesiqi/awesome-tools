#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, json, time
from datetime import timedelta

from spb import SPB

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

    # == == == == == == == == Part 2: Identify awakening time and falling time == == == == == == == == #
    start_time = time.time()

    print('>>> Identify awakening time and falling time for video: {0}'.format(vid))

    spb_model = SPB(data=daily_view)
    spb_model.identify_awakening_and_falling()
    print('>>> Total running time: {0}'.format(str(timedelta(seconds=time.time() - start_time)))[:-3])
    spb_model.plot_func('YouTubeID={0}'.format(vid))
