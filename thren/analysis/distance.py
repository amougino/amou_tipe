import sim
import numpy as np


def distance(solution):

    x_sat = np.array(solution.y[0])
    y_sat = np.array(solution.y[1])
    d = sim.pyth(x_sat, y_sat)

    return d


def same_d_reached(solution):
    d = distance(solution)
    start_d = d[0]
    if start_d > d[-1]:
        raise Exception("start distance greater than final")
    idx_same_d = len(d) - 1
    while start_d < d[idx_same_d]:
        idx_same_d -= 1
    if idx_same_d == 0:
        raise Exception("no final distance found")
    return idx_same_d


def min_b_dist(solution, settings):
    x_sat = np.array(solution.y[0])
    y_sat = np.array(solution.y[1])
    b1x, b1y, b2x, b2y = sim.body_positions(settings, solution.t)
    b1_dist = sim.pyth(x_sat - b1x, y_sat - b1y)
    b2_dist = sim.pyth(x_sat - b2x, y_sat - b2y)
    return min(b1_dist), min(b2_dist)
