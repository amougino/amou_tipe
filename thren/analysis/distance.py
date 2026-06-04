import sim
import numpy as np


def distance(solution):

    x_sat = np.array(solution.y[0])
    y_sat = np.array(solution.y[1])
    d = sim.pyth(x_sat, y_sat)

    return d
