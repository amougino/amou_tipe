'''

Satellite energy

'''
import sim as sim
import numpy as np

import matplotlib.pyplot as plt

settings = sim.get_single_settings()

n_of_points = 5000

end_time = sim.end_time_approx(settings)
timespan = (0, end_time)
time_values = np.linspace(0, end_time, n_of_points)

a = sim.calculate(settings, timespan, time_values=time_values, method="DOP853")

x_sat = np.array(a.y[0])
y_sat = np.array(a.y[1])
vx_sat = np.array(a.y[2])
vy_sat = np.array(a.y[3])
v_sat = sim.pyth(vx_sat, vy_sat)

t = np.array(a.t)

R1x, R1y, R2x, R2y = sim.body_positions(settings, t)

d1 = sim.pyth(x_sat - R1x, y_sat - R1y)
d2 = sim.pyth(x_sat - R2x, y_sat - R2y)

print(min(d1), min(d2))

e = (0.5*(v_sat**2)) - ((sim.G*settings["mass1"])/d1) - ((sim.G*settings["mass2"])/d2)

fig, ax = plt.subplots()
ax.plot(t, e)
plt.show()
