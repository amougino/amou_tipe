'''

THREAD THE NEEDLE

Trajectory maneuver

'''

import os
import json
import numpy as np
from scipy.integrate import solve_ivp as solve
import matplotlib.pyplot as plt
import matplotlib.animation as animation

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

G = 6.673 * 10**(-11)


def pyth(a, b):
    return np.sqrt(a**2 + b**2)


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


def to_bin_sys(settings):
    '''
    Get satellite speed in the binary system's frame of reference,
    from the original frame of reference.
    '''
    return (
        settings["sat_vel"]["x"] - settings["bin_sys_vel"]["x"],
        settings["sat_vel"]["y"] - settings["bin_sys_vel"]["y"]
    )


def from_bin_sys(vx, vy, settings):
    '''
    Get satellite speed in original frame of reference,
    from the binary system's frame of reference.
    '''
    return (
        vx + settings["bin_sys_vel"]["x"],
        vy + settings["bin_sys_vel"]["y"]
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


def calculate(settings, timespan, method="RK45", time_values=None):
    vx, vy = to_bin_sys(settings)
    initial = [
        settings["sat_pos"]["x"],
        settings["sat_pos"]["y"],
        vx,
        vy
    ]
    ds = define_ode(settings)
    return solve(fun=ds, t_span=timespan, y0=initial, method=method, rtol=1e-8, atol=1e-8, t_eval=time_values)


def end_time(settings, factor=1.5):
    dist_to_center = pyth(settings["sat_pos"]["x"], settings["sat_pos"]["y"])
    velocity = pyth(settings["sat_vel"]["x"], settings["sat_vel"]["y"])
    return (dist_to_center / velocity) * factor


def plot_traj(solution, settings, timespan, precision=10):

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

    plt.plot(solution.y[0], solution.y[1], label="satellite", c="blue")
    plt.plot(R1x, R1y, label="body 1", c="orange")
    plt.plot(R2x, R2y, label="body 2", c="green")

    ax.legend()

    plt.show()

    fig.savefig("fronde_deux_corps_amouginot.pdf")


def animate_traj(solution, settings, size, time_values, simulation_time=1):

    m1 = settings["mass1"]
    m2 = settings["mass2"]
    r = settings["body_distance"]

    m, omega = parameters(m1, m2, r)

    fig, ax = plt.subplots()

    iterations = len(solution.y[0])

    scat_sat = ax.scatter(solution.y[0][0], solution.y[1][0], c="blue")
    scat_b1 = ax.scatter((-m/m1) * r, 0, c="orange")
    scat_b2 = ax.scatter((m/m2) * r, 0, c="green")
    ax.set(xlim=[-size, size], ylim=[-size, size])

    def update(frame):
        sat_x = solution.y[0][frame]
        sat_y = solution.y[1][frame]

        t = time_values[frame]

        Rx = r*np.cos(omega*t)
        Ry = r*np.sin(omega*t)

        R1x = (-m/m1) * Rx
        R1y = (-m/m1) * Ry
        R2x = (m/m2) * Rx
        R2y = (m/m2) * Ry

        sat_data = np.stack([sat_x, sat_y]).T
        scat_sat.set_offsets(sat_data)

        b1_data = np.stack([R1x, R1y]).T
        scat_b1.set_offsets(b1_data)
        b2_data = np.stack([R2x, R2y]).T
        scat_b2.set_offsets(b2_data)

        return (scat_sat)

    print(iterations)
    interval = (simulation_time / iterations) * 1000

    ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=interval)
    plt.show()
