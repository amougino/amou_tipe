'''

Parallel version of on_the_fly

'''
import thren.sim as sim
import os
import itertools
import json
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from thren.create_folder import create_folder
import thren.sim_analysis.transfer as transfer
import thren.sim_analysis.distance as distance

'''
IMPORTANT :
the call to parallel_otf must be under 'if __name__ == "__main__":'
'''

NONE_LINE = "None,None,None,None,None"


def resolve_workers(workers):
    '''
    number of processes to use, from :
        None        -> every logical cpu
        int  n      -> exactly n processes                  ex: 4
        float f     -> fraction of the cpus, 0 < f <= 1     ex: 0.5  (half of the cpus)
        str "p%"    -> percentage of the cpus               ex: "50%"
    always at least 1 and at most cpu_count()
    '''
    n_cpu = cpu_count()

    if workers is None:
        return n_cpu

    if isinstance(workers, str):
        text = workers.strip()
        if not text.endswith("%"):
            raise ValueError(f'workers string must look like "50%", got {workers!r}')
        percent = float(text[:-1])
        if not 0 < percent <= 100:
            raise ValueError(f"workers percentage must be in ]0, 100], got {workers!r}")
        return max(1, min(n_cpu, round(n_cpu * percent / 100)))

    if isinstance(workers, float):
        if not 0 < workers <= 1:
            raise ValueError(f"workers fraction must be in ]0, 1], got {workers}")
        return max(1, min(n_cpu, round(n_cpu * workers)))

    if isinstance(workers, int) and not isinstance(workers, bool):
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        return min(n_cpu, workers)

    raise TypeError(f"workers must be None, int, float or str, got {type(workers).__name__}")


def parallel_otf(settings_file, workers=None, chunksize=16):
    '''
    workers : number of logical cpus
    chunksize : number of cells sent to each worker at a time
    '''
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

    directory = os.path.join(os.path.dirname(os.path.abspath(settings_file)), "results")
    os.makedirs(directory, exist_ok=True)
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

    variations = multi_settings["variations"]
    names = [p["name"] for p in variations]
    grids = [np.linspace(p["start"], p["stop"], p["points"]) for p in variations]

    result_file = result_directory + "/data.csv"
    with open(result_file, 'w') as f:
        f.write(",".join(names) + ",v,e,theta,min_b1,min_b2\n")

    total = int(np.prod([len(g) for g in grids]))
    workers = resolve_workers(workers)
    jobs = ((settings, names, combo) for combo in itertools.product(*grids))

    print(f"{total} cells on {workers}/{cpu_count()} workers -> {result_directory}")
    start = time.time()

    with Pool(workers) as pool, open(result_file, 'a') as f:
        for done, line in enumerate(pool.imap(_cell, jobs, chunksize=chunksize), start=1):
            f.write(line)
            if done % 100 == 0 or done == total:
                f.flush()
                elapsed = time.time() - start
                print(f"{done}/{total}  {elapsed:.0f}s  (~{elapsed / done * (total - done):.0f}s left)")

    print(f"parallel otf done in {time.time() - start:.0f}s")
    return result_directory


def _cell(args):
    '''
    simulates and analyses a single cell, returns one csv line
    '''
    settings, names, combo = args
    s = dict(settings)
    for name, value in zip(names, combo):
        s[name] = value

    head = ",".join(str(i) for i in combo)

    try:
        timespan = (0, sim.end_time_approx(s))
        simulation = sim.calculate(s, timespan)
        if simulation.status != 0:
            # -1 : solver failed
            # 1 : terminal event (ex: collision)
            return f"{head},{NONE_LINE}\n"

        v = transfer.vel_transfer(simulation, s)
        e = transfer.energy_transfer(simulation, s)
        theta = transfer.Delta_theta(simulation)
        min_b1, min_b2 = distance.min_b_dist(simulation, s)
        return f"{head},{v},{e},{theta},{min_b1},{min_b2}\n"
    except Exception:
        return f"{head},{NONE_LINE}\n"
