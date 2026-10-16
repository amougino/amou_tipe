'''

THREAD THE NEEDLE
ThreN

Trajectory maneuver

'''

import os
import json
import numpy as np
from scipy.integrate import solve_ivp as solve
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from thren.formulae import *

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def get_settings(file=os.path.join(__location__, "settings_single.json")):
    '''
    returns dict of json file
    '''
    settings = {}
    with open(file) as f:
        settings = json.load(f)
    return settings


def to_bin_sys(settings):
    '''
    gets satellite speed in the binary system's frame of reference, from the original frame of reference
    '''
    return (
        settings["sat_vel_x"] - settings["bin_sys_vel_x"],
        settings["sat_vel_y"] - settings["bin_sys_vel_y"]
    )


def from_bin_sys(vx, vy, settings):
    '''
    gets satellite speed in original frame of reference, from the binary system's frame of reference
    '''
    return (
        vx + settings["bin_sys_vel_x"],
        vy + settings["bin_sys_vel_y"]
    )


def to_synodique(x, y, theta):
    '''
    changes a position vector in original frame of reference to a vector in a rotated frame of reference
    '''
    x0 = x*np.cos(theta) + y*np.sin(theta)
    y0 = - x*np.sin(theta) + y*np.cos(theta)
    return x0, y0


def to_synodique_vel(vel_x, vel_y, omega, t, x, y):
    '''
    changes a velocity vector in original frame of reference to a vector in a rotated frame of reference
    '''
    xp, yp = to_synodique(x, y, omega*t)
    vx0 = vel_x*np.cos(omega*t) + vel_y*np.sin(omega*t) + omega*yp
    vy0 = - vel_x*np.sin(omega*t) + vel_y*np.cos(omega*t) - omega*xp
    return vx0, vy0


def collision_events(settings, radius=1e6):
    m1, m2, r = settings["mass1"], settings["mass2"], settings["body_distance"]
    m, omega = parameters(m1, m2, r)

    def hit1(t, s):
        return np.hypot(s[0] + (m/m1)*r*np.cos(omega*t),
                        s[1] + (m/m1)*r*np.sin(omega*t)) - radius

    def hit2(t, s):
        return np.hypot(s[0] - (m/m2)*r*np.cos(omega*t),
                        s[1] - (m/m2)*r*np.sin(omega*t)) - radius

    hit1.terminal = hit2.terminal = True
    return [hit1, hit2]


def calculate(settings, timespan, method="DOP853", time_values=None):
    '''
    returns the solved ordinary differential equation
    '''
    vx, vy = to_bin_sys(settings)
    initial = [
        settings["sat_pos_x"],
        settings["sat_pos_y"],
        vx,
        vy
    ]
    ds = define_ode(settings)
    m1 = settings["mass1"]
    m2 = settings["mass2"]
    r = settings["body_distance"]
    m, omega = parameters(m1, m2, r)
    return solve(
        fun=ds,
        t_span=timespan,
        y0=initial,
        method=method,
        rtol=1e-12, atol=1e-12,
        t_eval=time_values,
        max_step=(2 * np.pi) / (1000*omega),
        events=collision_events(settings, 1e6))


def body_positions(settings, t):
    '''
    returns the positions of the two bodies and a point in time t
    '''
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

    return R1x, R1y, R2x, R2y


def plot_traj(solution, settings, timespan, precision=10, save=False):
    '''
    plots trajectory of the bodies and of the satellite using the solved ode
    '''
    start, stop = timespan
    t = np.linspace(start, stop, precision)

    R1x, R1y, R2x, R2y = body_positions(settings, t)

    fig, ax = plt.subplots()

    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")

    plt.plot(solution.y[0], solution.y[1], label="satellite", c="blue")
    plt.plot(R1x, R1y, label="body 1", c="orange")
    plt.plot(R2x, R2y, label="body 2", c="green")

    ax.legend()
    ax.axis('equal')

    plt.show()

    if save:
        fig.savefig("fronde_deux_corps_amouginot.pdf")


def animate_traj(solution, settings, size, time_values, simulation_time=1, trail_fraction=10, save=False):
    '''
    animates trajectory of the bodies and of the satellite using the solved ode
    '''
    m1 = settings["mass1"]
    m2 = settings["mass2"]
    r = settings["body_distance"]

    m, omega = parameters(m1, m2, r)

    fig, ax = plt.subplots()

    iterations = len(solution.y[0])

    scat_sat = ax.scatter(solution.y[0][0], solution.y[1][0], c="blue")
    scat_b1 = ax.scatter((-m/m1) * r, 0, c="orange")
    scat_b2 = ax.scatter((m/m2) * r, 0, c="green")

    trail_length = iterations // trail_fraction
    scat_sat_trail = ax.scatter(0, 0, c="black", s=1)
    trail = [[size], [size]]

    ax.set(xlim=[-size, size], ylim=[-size, size])

    def update(frame):

        if frame == 0:
            trail[0] = [size]
            trail[1] = [size]

        sat_x = solution.y[0][frame]
        sat_y = solution.y[1][frame]

        t = time_values[frame]

        R1x, R1y, R2x, R2y = body_positions(settings, t)

        sat_data = np.stack([sat_x, sat_y]).T
        scat_sat.set_offsets(sat_data)
        b1_data = np.stack([R1x, R1y]).T
        scat_b1.set_offsets(b1_data)
        b2_data = np.stack([R2x, R2y]).T
        scat_b2.set_offsets(b2_data)
        trail_data = np.stack(trail).T
        scat_sat_trail.set_offsets(trail_data)

        trail[0] += [sat_x]
        trail[1] += [sat_y]
        removed = max(0, len(trail[0]) - trail_length)
        trail[0] = trail[0][removed:]
        trail[1] = trail[1][removed:]

        return (scat_sat)

    interval = (simulation_time / iterations) * 1000

    ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=interval)

    if save:
        writer = animation.PillowWriter(
            fps=30,
            metadata=dict(artist='amougino'),
            bitrate=1800
        )
        ani.save(__location__ + "/animation.gif", writer=writer)

    plt.show()
