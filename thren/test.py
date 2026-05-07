import thren_v_2 as thren
import numpy as np

import matplotlib.pyplot as plt


settings = thren.get_single_settings()

n_of_points = 1000

end_time = thren.end_time_approx(settings)
timespan = (0, end_time)
time_values = np.linspace(0, end_time, n_of_points)

a = thren.calculate(settings, timespan, time_values=time_values)

# thren.plot_traj(a, settings, timespan)

thren.animate_traj(a, settings, abs(settings["sat_pos"]["y"]), time_values, simulation_time=5, trail_fraction=10)
