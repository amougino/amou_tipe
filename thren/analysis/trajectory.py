import sim
import numpy as np
import matplotlib.pyplot as plt


def trajectory_zoomed(solution, settings, distance):

    x_sat = np.array(solution.y[0])
    y_sat = np.array(solution.y[1])

    d = sim.pyth(x_sat, y_sat)
    valid_idxs = [i for i in range(len(d)) if d[i] <= distance]

    new_x = [x_sat[i] for i in valid_idxs]
    new_y = [y_sat[i] for i in valid_idxs]

    t = np.array(solution.t)

    b1x, b1y, b2x, b2y = sim.body_positions(settings, np.array([t[i] for i in valid_idxs]))

    return new_x, new_y, b1x, b1y, b2x, b2y
