import numpy as np
import analysis.velocity as velocity
import analysis.sat_energy as sat_energy
import analysis.distance as distance


def vel_transfer(solution, settings):
    v = velocity.velocity(solution, settings)
    final_v_idx = distance.same_d_reached(solution)
    return v[final_v_idx]/v[0]


def energy_transfer(solution, settings):
    e = sat_energy.sat_energy(solution, settings)
    final_v_idx = distance.same_d_reached(solution)
    return e[final_v_idx]/e[0]


def theta(x1, y1, x2, y2):
    return np.atan2(x2 - x1, y2 - y1)


def Delta_theta(solution):
    start = theta(
        solution.y[0][1],
        solution.y[1][1],
        solution.y[0][0],
        solution.y[1][0]
    )
    final_v_idx = distance.same_d_reached(solution)
    end = theta(
        solution.y[0][final_v_idx],
        solution.y[1][final_v_idx],
        solution.y[0][final_v_idx + 1],
        solution.y[1][final_v_idx + 1]
    )
    return (start - end) % (2*np.pi)
