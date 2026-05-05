import thren_v_2 as thren
import numpy as np

import matplotlib.pyplot as plt


settings = thren.get_single_settings()

n_of_points = 100000

end_time = thren.end_time(settings)
timespan = (0, end_time)
time_values = np.linspace(0, end_time, n_of_points)

a = thren.calculate(settings, timespan, time_values=time_values)

# thren.plot_traj(a, settings, timespan)

fig, ax = plt.subplots()

v = []
v2 = []
for i in range(n_of_points):
    vx, vy = thren.from_bin_sys(a.y[2][i], a.y[3][i], settings)
    v.append(thren.pyth(vx, vy))
    v2.append(thren.pyth(a.y[2][i], a.y[3][i]))

ax.grid()
ax.plot([i for i in range(n_of_points)], v)
ax.plot([i for i in range(n_of_points)], v2)

thren.animate_traj(a, settings, abs(settings["sat_pos"]["y"]), time_values, simulation_time=5, trail_fraction=10)
