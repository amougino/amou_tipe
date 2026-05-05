'''

w.i.p.

'''

import thren_v_2 as thren
import numpy as np

import matplotlib.pyplot as plt

settings = thren.get_single_settings()

n_of_points = 10000

end_time = thren.end_time_approx(settings)
timespan = (0, end_time)
time_values = np.linspace(0, end_time, n_of_points)

a = thren.calculate(settings, timespan, time_values=time_values)

x_sat = np.array(a.y[0])
y_sat = np.array(a.y[1])
vx_sat = np.array(a.y[2])
vy_sat = np.array(a.y[3])

start, stop = timespan
t = np.linspace(start, stop, 10)

m1 = settings["mass1"]
m2 = settings["mass2"]
r = settings["body_distance"]

m, omega = thren.parameters(m1, m2, r)

Rx = r*np.cos(omega*t)
Ry = r*np.sin(omega*t)

R1x = (-m/m1) * Rx
R1y = (-m/m1) * Ry
R2x = (m/m2) * Rx
R2y = (m/m2) * Ry
