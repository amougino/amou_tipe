'''

Satellite energy

'''
import sim
import numpy as np


def sat_energy(solution, settings):

    x_sat = np.array(solution.y[0])
    y_sat = np.array(solution.y[1])
    vx_sat = np.array(solution.y[2])
    vy_sat = np.array(solution.y[3])
    v_sat = sim.pyth(vx_sat, vy_sat)

    t = np.array(solution.t)

    R1x, R1y, R2x, R2y = sim.body_positions(settings, t)

    d1 = sim.pyth(x_sat - R1x, y_sat - R1y)
    d2 = sim.pyth(x_sat - R2x, y_sat - R2y)

    e = (0.5*(v_sat**2)) - ((sim.G*settings["mass1"])/d1) - ((sim.G*settings["mass2"])/d2)

    return e
