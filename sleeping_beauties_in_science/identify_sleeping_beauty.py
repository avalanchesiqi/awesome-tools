"""
Identify revival time and hibernation time in time-series
algorithm from: "Defining and identifying Sleeping Beauties in science" PNAS'15
and "Sleeping beauties in meme diffusion"
"""

import json, time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt


def dist_and_proj(line_params, point_coordination):
    """ |ax_0 + by_0 + c| / sqrt(a^2 + b^2) for line ax+by+c=0 and point (x_0, y_0)
    source: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    """
    a, b, c = line_params
    x_0, y_0 = point_coordination
    # print('coordination', point_coordination, 'distance', abs(a*x_0 + b*y_0 + c) / np.sqrt(a**2 + b**2), 'intersect', ((-a*b*y_0 + b**2*x_0 - a*c) / (a**2 + b**2), (a**2*y_0 - a*b*x_0 - b*c) / (a**2 + b**2)))
    return abs(a*x_0 + b*y_0 + c) / np.sqrt(a**2 + b**2), ((-a*b*y_0 + b**2*x_0 - a*c) / (a**2 + b**2), (a**2*y_0 - a*b*x_0 - b*c) / (a**2 + b**2))


def identify_revival_and_hibernation(ts_data, start, end):
    ts_data = np.array(ts_data)
    t_peak = np.argmax(ts_data[start: end])
    # print(t_peak, ts_data[t_peak])
    grow_reference_line = lambda t: (ts_data[t_peak] - ts_data[start]) / (t_peak - start) * (t - start) + ts_data[start]
    # fade_reference_line = lambda t: (ts_data[end] - ts_data[t_peak]) / (end - t_peak) * (t - t_peak) + ts_data[t_peak]
    grow_reference_line_params = [(ts_data[t_peak] - ts_data[start]) / (t_peak - start), -1,
                                  ts_data[start] - (ts_data[t_peak] - ts_data[start]) / (t_peak - start) * start]
    fade_reference_line_params = [(ts_data[end] - ts_data[t_peak]) / (end - t_peak), -1,
                                  ts_data[t_peak] - (ts_data[end] - ts_data[t_peak]) / (end - t_peak) * t_peak]
    # time of revival, argmax distance to grow reference line
    t_revival = np.argmax([dist_and_proj(grow_reference_line_params, (x_0, y_0))[0]
                           for x_0, y_0 in enumerate(ts_data[start: t_peak])])
    t_hibernation = t_peak + np.argmax([dist_and_proj(fade_reference_line_params, (t_peak + x_0, y_0))[0]
                                        for x_0, y_0 in enumerate(ts_data[t_peak: end])])

    b_coefficient = np.sum([(grow_reference_line(i) - ts_data[i])/max(1, i) for i in range(start, t_peak)])

    proj_revival_x, proj_revival_y = dist_and_proj(grow_reference_line_params, (t_revival, ts_data[t_revival]))[1]
    proj_hibernation_x, proj_hibernation_y = dist_and_proj(fade_reference_line_params, (t_hibernation, ts_data[t_hibernation]))[1]

    return start, [(t_revival, t_peak, t_hibernation, b_coefficient, (proj_revival_x, proj_revival_y), (proj_hibernation_x, proj_hibernation_y))], end


if __name__ == '__main__':
    start_time = time.time()

    # == == == == == == Part 1: load time series data == == == == == == #
    with open('data/sleep_beauties.json', 'r') as fin:
        videos_str = fin.readlines()
        days = json.loads(videos_str[0])['insights']['days']
        views = json.loads(videos_str[0])['insights']['dailyView']
        # paddle zero in missing days
        daily_view = [0]*(days[-1] + 1)
        for i in range(len(days)):
            daily_view[days[i]] = views[i]
        daily_view = np.array(daily_view)
        scale = np.max(daily_view) / days[-1]
        norm_daily_view = daily_view / scale

    # == == == == == == Part 2: identify revival and hibernation time == == == == == == #
    start, cycles, end = identify_revival_and_hibernation(norm_daily_view, start=0, end=len(norm_daily_view)-1)
    print(start, cycles, end)

    # == == == == == == Part 2: visualize time series data == == == == == == #
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)

    ax1.plot(np.arange(len(daily_view)), daily_view, 'k')
    for cycle_idx, cycle in enumerate(cycles):
        if cycle_idx == 0:
            t_revival, t_peak, t_hibernation, b_coefficient, (proj_rx, proj_ry), (proj_hx, proj_hy) = cycle
            b_coefficient = b_coefficient * scale
            ax1.plot([start, t_peak], [daily_view[start], daily_view[t_peak]], 'r--')
            ax1.scatter(t_revival, daily_view[t_revival], marker='o', color='r', zorder=30)
            ax1.plot([t_revival, proj_rx], [daily_view[t_revival], proj_ry * scale], 'r--')
            ax1.text(0.03, 0.9, 'B coefficient={0:.2f}\n(revival, peak, hibernation)={1}, {2}, {3}'.format(b_coefficient, t_revival, t_peak, t_hibernation),
                     transform=ax1.transAxes)
        if cycle_idx == len(cycles) - 1:
            t_revival, t_peak, t_hibernation, b_coefficient, (proj_rx, proj_ry), (proj_hx, proj_hy) = cycle
            ax1.plot([t_peak, end], [daily_view[t_peak], daily_view[end]], 'r--')
            ax1.scatter(t_hibernation, daily_view[t_hibernation], marker='o', color='r', zorder=30)
            ax1.plot([t_hibernation, proj_hx], [daily_view[t_hibernation], proj_hy * scale], 'r--')
    ax1.set_xlabel('video age')
    ax1.set_ylabel('daily view')
    ax1.tick_params(axis='y', rotation=90)

    plt.tight_layout()
    plt.show()
