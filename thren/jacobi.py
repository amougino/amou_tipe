'''

Jacobi constant

'''
import sim as sim
import numpy as np
import matplotlib.pyplot as plt


def jacobi(solution, settings):
    '''
    returns jacobi constant over time for a solved ode
    '''
    x_sat = np.array(solution.y[0])
    y_sat = np.array(solution.y[1])
    vx_sat = np.array(solution.y[2])
    vy_sat = np.array(solution.y[3])

    t = np.array(solution.t)

    m1 = settings["mass1"]
    m2 = settings["mass2"]
    r = settings["body_distance"]

    m, omega = sim.parameters(m1, m2, r)

    R1x, R1y, R2x, R2y = sim.body_positions(settings, t)

    d1 = sim.pyth(x_sat - R1x, y_sat - R1y)
    d2 = sim.pyth(x_sat - R2x, y_sat - R2y)

    x_syn, y_syn = sim.to_synodique(x_sat, y_sat, omega*t)
    x_vel_syn, y_vel_syn = sim.to_synodique_vel(vx_sat, vy_sat, omega, t, x_sat, y_sat)

    mu1 = sim.G * settings["mass1"]
    mu2 = sim.G * settings["mass2"]

    j = ((omega**2)*(x_syn**2 + y_syn**2)) + (2*((mu1/d1) + (mu2/d2))) - (x_vel_syn**2 + y_vel_syn**2)

    return j
