import thren_v_1 as thren
import numpy as np


settings = thren.get_single_settings()

print('initial vel :', thren.pyth(settings["sat_vel"]["x"], settings["sat_vel"]["y"]))

n_of_points = 100

end_time = thren.end_time(settings)
timespan = (0, end_time)
time_values = np.linspace(0, end_time, n_of_points)

a = thren.calculate(settings, timespan, time_values=time_values)

# thren.plot_traj(a, settings, timespan)

vfx, vfy = thren.from_bin_sys(a.y[2][-1], a.y[3][-1], settings)
print('final vel : ', thren.pyth(vfx, vfy))

thren.animate_traj(a, settings, abs(settings["sat_pos"]["y"]), time_values)
