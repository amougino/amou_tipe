'''

Create a csv file with analysed data

'''
import sim
import os
import copy
import numpy as np
from create_folder import create_folder
import analysis.transfer as transfer
import analysis.distance as distance
import json


def on_the_fly(settings_file):
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
    for parameter in multi_settings["constants"]:
        settings[parameter] = multi_settings["constants"][parameter]

    directory = os.path.dirname(settings_file) + "/results"
    files_in_dir = os.listdir(directory)
    number = 1
    for file in files_in_dir:
        num = file.split("_otf_")
        if len(num) != 1:
            number = max(number, int(num[1]) + 1)
    result_directory = directory + "/_otf_" + str(number)
    create_folder(result_directory)

    constants_file = result_directory + "/constants.json"
    with open(constants_file, 'w') as f:
        json.dump(multi_settings["constants"], f, indent=4)

    result_file = result_directory + "/data.csv"
    header = multi_settings["variations"][0]["name"]
    for parameter in multi_settings["variations"][1:]:
        header += ',' + parameter["name"]
    header += ',v,e,theta,min_b1,min_b2\n'
    with open(result_file, 'w') as f:
        f.write(header)
        f.close()

    _iterate(result_file, multi_settings["variations"], settings)


def _iterate(file, remaining, settings, current=""):
    remaining_copy = copy.deepcopy(remaining)
    parameter = remaining_copy[0]
    remaining_copy.pop(0)
    empty = len(remaining_copy) == 0
    start, stop, num = parameter["start"], parameter["stop"], parameter["points"]
    values = np.linspace(start, stop, num)

    if empty:
        lines = []
        for i in values:
            settings[parameter["name"]] = i
            timespan = (0, sim.end_time_approx(settings))
            simulation = sim.calculate(settings, timespan)
            if simulation.status == -1:
                to_add = f"{i},None,None,None,None,None\n"
                line = to_add if current == "" else current + ',' + to_add
                lines.append(line)
            else:
                v = transfer.vel_transfer(simulation, settings)
                e = transfer.energy_transfer(simulation, settings)
                theta = transfer.Delta_theta(simulation)
                min_b1, min_b2 = distance.min_b_dist(simulation, settings)
                to_add = f"{i},{v},{e},{theta},{min_b1},{min_b2}\n"
                line = to_add if current == "" else current + ',' + to_add
                lines.append(line)
        with open(file, 'a') as f:
            f.writelines(lines)
    else:
        for i in values:
            new_current = str(i) if current == "" else current + ',' + str(i)
            settings[parameter["name"]] = i
            _iterate(file, remaining_copy, settings, new_current)
