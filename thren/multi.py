import os
import numpy as np
import sim
import copy


def multi(settings_file=os.path.join(sim.__location__, "settings_multi.json")):

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

    return _iterate(multi_settings["variations"], settings)


def _iterate(remaining, settings):
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
