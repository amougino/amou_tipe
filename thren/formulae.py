import math
import numpy as np


G = 6.673 * 10**(-11)


def pyth(a, b):
    '''
    returns the hypotenuse of a right angle triangle of sides a and b
    '''
    return np.sqrt(a**2 + b**2)


def parameters(m1, m2, r):
    '''
    returns parameters defining the movement of the two planets
    '''
    m = (m1 * m2) / (m1 + m2)
    # omega = (2/r) * np.sqrt((G*m)/r) ##### testing different omega #####
    omega = np.sqrt(G * (m1 + m2) / r**3)
    return (
        m,
        omega
    )


def end_time_approx(settings, factor=1.8):
    '''
    returns time needed to cross to the other side of the origin in an empty system, multiplied by a factor
    '''
    dist_to_center = pyth(settings["sat_pos_x"], settings["sat_pos_y"])
    velocity = pyth(settings["sat_vel_x"], settings["sat_vel_y"])
    return (2*dist_to_center / velocity) * factor


def define_ode(settings):
    m1, m2, r = settings["mass1"], settings["mass2"], settings["body_distance"]
    m, omega = parameters(m1, m2, r)
    R1, R2 = m*r/m1, m*r/m2          # body offsets from the barycenter, per unit of cos/sin
    Gm1, Gm2 = G*m1, G*m2

    def ds(t, s):
        x, y, vx, vy = s
        c, sn = math.cos(omega*t), math.sin(omega*t)
        dx1, dy1 = x + R1*c, y + R1*sn
        dx2, dy2 = x - R2*c, y - R2*sn
        q1 = dx1*dx1 + dy1*dy1
        q2 = dx2*dx2 + dy2*dy2
        k1 = Gm1 / (q1 * math.sqrt(q1))
        k2 = Gm2 / (q2 * math.sqrt(q2))
        return [vx, vy, -k1*dx1 - k2*dx2, -k1*dy1 - k2*dy2]

    return ds


def arrival_time(x0, v0, m):
    '''
    if considering the two planets as a single, motionless planet of mass m = m1 + m2,
    time of arrival at the center of the planet
    leaving from a position x0 away, moving at velocity v0 towards it

    works only if
    v0^2 > 2*G*m/x0
    '''
    tf = (
        (v0*(x0**2)) / ((v0**2)*x0 - 2*G*m) -
        (
            ((2*G*m*(x0**(3/2))) / (((v0**2)*x0 - 2*G*m)**(3/2))) *
            np.arcsinh((((v0**2)*x0-2*G*m) / (2*G*m))**(1/2))
        )
    )
    return tf


def arrival_phase_unwrapped(settings):
    m1 = np.asarray(settings["mass1"], dtype=float)
    m2 = np.asarray(settings["mass2"], dtype=float)
    r = np.asarray(settings["body_distance"], dtype=float)
    m, omega = parameters(m1, m2, r)

    x0 = -np.asarray(settings["sat_pos_y"], dtype=float)
    v0 = np.asarray(settings["sat_vel_y"], dtype=float)
    mass = m1 + m2

    valid = (x0 > 0) & ((v0**2) * x0 > 2 * G * mass)       # arrival_time only exists then
    tf = arrival_time(np.where(valid, x0, 1.0), np.where(valid, v0, 1.0), mass)
    return np.where(valid, tf * omega, np.nan)


def arrival_phase(settings):
    return arrival_phase_unwrapped(settings) % (2 * np.pi)
