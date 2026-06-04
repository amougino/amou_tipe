'''

Same as multi, but stores in files

'''
import os
import numpy as np
import sim
import copy


def create_folder(path):
    try:
        os.mkdir(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


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

    directory = os.path.dirname(settings_file)
    files_in_dir = os.listdir(directory)
    number = 1
    for file in files_in_dir:
        num = file.split("_sim_")
        if len(num) != 1:
            number = max(number, int(num[1]) + 1)
    create_folder(directory + "/_sim_" + str(number))
    print(files_in_dir)

    for parameter in multi_settings["constants"]:
        settings[parameter] = multi_settings["constants"][parameter]

    print(settings)

    # return _iterate(multi_settings["variations"], settings)
populate(sim.__location__ + "/settings_multi.json")


def _iterate(remaining, settings):
    print(1, remaining)
    remaining_copy = copy.deepcopy(remaining)
    parameter = remaining_copy[0]
    remaining_copy.pop(0)
    empty = len(remaining_copy) == 0
    start, stop, num = parameter["start"], parameter["stop"], parameter["points"]
    values = np.linspace(start, stop, num)
    results = []
    if empty:
        for i in values:
            settings[parameter["name"]] = i
            timespan = (0, sim.end_time_approx(settings))
            simulation = sim.calculate(settings, timespan)
            results.append(simulation)
    else:
        for i in values:
            settings[parameter["name"]] = i
            simulations = _iterate(remaining_copy, settings)
            results.append(simulations)
    return results
