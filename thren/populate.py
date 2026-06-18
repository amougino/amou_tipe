'''

Same as multi, but stores in files

'''
import os
import numpy as np
import sim
import copy
import json
import pandas as pd
from create_folder import create_folder

import time


def populate(settings_file):

    multi_settings = sim.get_settings(settings_file)

    settings = {
        "mass1": None,
        "mass2": None,
        "body_distance": None,
        "sat_pos_x": None,
        "sat_pos_y": None,
        "sat_vel_x": None,
        "sat_vel_y": None,
        "bin_sys_vel_x": None,
        "bin_sys_vel_y": None
    }

    directory = os.path.dirname(settings_file) + "/results"
    files_in_dir = os.listdir(directory)
    number = 1
    for file in files_in_dir:
        num = file.split("_sim_")
        if len(num) != 1:
            number = max(number, int(num[1]) + 1)
    result_directory = directory + "/_sim_" + str(number)
    create_folder(result_directory)

    for parameter in multi_settings["constants"]:
        settings[parameter] = multi_settings["constants"][parameter]

    _iterate_save(multi_settings["variations"], settings, result_directory)


def _iterate_save(remaining, settings, location):
    remaining_copy = copy.deepcopy(remaining)

    with open(location + "/settings.json", 'w') as f:
        json.dump(settings, f, indent=4)

    parameter = remaining_copy[0]
    remaining_copy.pop(0)
    empty = len(remaining_copy) == 0
    start, stop, num = parameter["start"], parameter["stop"], parameter["points"]
    values = np.linspace(start, stop, num)

    if empty:
        for i in values:
            start = time.time()
            settings[parameter["name"]] = i
            timespan = (0, sim.end_time_approx(settings))
            simulation = sim.calculate(settings, timespan)
            df = pd.DataFrame({
                't': simulation.t,
                'x': simulation.y[0],
                'y': simulation.y[1],
                'vx': simulation.y[2],
                'vy': simulation.y[3],
            })
            df.to_csv(location + f"/{parameter["name"]}_{i}.csv", index=False)
            print(time.time() - start, parameter["name"], i)
    else:
        for i in values:
            print(parameter["name"], i)
            settings[parameter["name"]] = i
            result_directory = location + f"/{parameter["name"]}_{i}"
            create_folder(result_directory)
            _iterate_save(remaining_copy, settings, result_directory)


populate(sim.__location__ + "/settings_multi.json")
