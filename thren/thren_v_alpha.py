'''

THREAD THE NEEDLE

Trajectory maneuver

'''

import os
import json
import numpy as np
from scipy.integrate import solve_ivp as solve
import matplotlib.pyplot as plt


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

G = 6.673 * 10**(-11)


def get_single_settings(file=os.path.join(__location__, "settings_single.json")):
    settings = {}
    with open(file) as f:
        settings = json.load(f)
    return settings


def parameters(m1, m2, r):
    m = (m1 * m2) / (m1 + m2)
    return (
        m,
        (2/r) * np.sqrt((2*G*m)/r)
    )


def define_ode(settings):

    m1 = settings["mass1"]
    m2 = settings["mass2"]
    r = settings["body_distance"]

    m, omega = parameters(m1, m2, r)

    def ds(t, s):
        x = s[0]
        y = s[1]
        vx = s[2]
        vy = s[3]
        bx = m*r*np.cos(omega*t)
        by = m*r*np.sin(omega*t)
        d1 = np.sqrt(
            ((bx/m1) + x)**2 +
            ((by/m1) + y)**2
        )
        d2 = np.sqrt(
            ((bx/m2) - x)**2 +
            ((by/m2) - y)**2
        )
        return [
            vx,
            vy,
            G * ((-bx - m1*x) / (d1**3) + (bx - m2*x) / (d2**3)),
            G * ((-by - m1*y) / (d1**3) + (by - m2*y) / (d2**3))
        ]

    return ds


def calculate(settings, timespan, method="RK45"):
    initial = [
        settings["sat_pos"]["x"],
        settings["sat_pos"]["y"],
        settings["sat_vel"]["x"],
        settings["sat_vel"]["y"]
    ]
    ds = define_ode(settings)
    return solve(fun=ds, t_span=timespan, y0=initial, method=method, rtol=1e-8, atol=1e-8)


def plot_traj(solution, settings, timespan, precision):

    start, stop = timespan
    t = np.linspace(start, stop, precision)

    m1 = settings["mass1"]
    m2 = settings["mass2"]
    r = settings["body_distance"]

    m, omega = parameters(m1, m2, r)

    Rx = r*np.cos(omega*t)
    Ry = r*np.sin(omega*t)

    R1x = (-m/m1) * Rx
    R1y = (-m/m1) * Ry
    R2x = (m/m2) * Rx
    R2y = (m/m2) * Ry

    fig, ax = plt.subplots()

    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")

    plt.plot(solution.y[0], solution.y[1], label="satellite")
    plt.plot(R1x, R1y, label="body 1")
    plt.plot(R2x, R2y, label="body 2")

    ax.legend()

    plt.show()

    fig.savefig("fronde_deux_corps_amouginot.pdf")
