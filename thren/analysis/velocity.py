'''

Satellite velocity

'''
import sim
import numpy as np


def velocity(solution, settings):

    vx_sat = np.array(solution.y[2])
    vy_sat = np.array(solution.y[3])
    vx_new, vy_new = sim.from_bin_sys(vx_sat, vy_sat, settings)
    v_sat = sim.pyth(vx_new, vy_new)

    return v_sat
