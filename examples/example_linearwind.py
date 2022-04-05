import json
import os
import time

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

from mermoz.mdf_manager import MDFmanager
from mermoz.misc import *
from mermoz.params_summary import ParamsSummary
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.solver import Solver
from mermoz.trajectory import Trajectory
from mermoz.wind import LinearWind, DiscreteWind, UniformWind

mpl.style.use('seaborn-notebook')


def example_linear_wind():
    """
    Example of the PMP-based solver for the time-optimal Zermelo
    control problem with a linear windfield. Comparison to the
    analytical solution.
    """
    output_dir = '/home/bastien/Documents/work/mermoz/output/example_linearwind'
    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)

    # UAV airspeed in m/s
    v_a = 23.

    # The time window upper bound in seconds
    T = 14 * 3600

    x_f = 1e6
    x_target = np.array([x_f, 0.])

    # The wind gradient
    gradient = np.array([[0., v_a / 1e6],
                         [0., 0.]])
    origin = np.array([0., 0.])
    value_origin = np.array([0., 0.])

    # Wind array boundaries
    bl = 1e6 * np.array([-0.2, -1.])
    tr = 1e6 * np.array([1.2, 1.])

    linear_wind = LinearWind(gradient, origin, value_origin)
    total_wind = DiscreteWind()
    # Sample analytic linear wind to a grid
    total_wind.load_from_wind(linear_wind, 51, 51, bl, tr, 'cartesian')
    mdfm.dump_wind(total_wind)

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = np.array([0., 0.])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map', axes_equal=False)
    # mp.display.set_wind_density(2)

    nt_pmp = 10000

    t_start = time.time()
    solver = Solver(mp,
                    x_init,
                    x_target,
                    T,
                    -np.pi / 2. + 5e-2,
                    np.pi / 2. - 5e-2,
                    output_dir,
                    N_disc_init=2,
                    opti_ceil=1e6/30,
                    neighb_ceil=1e6/60,
                    n_min_opti=1,
                    adaptive_int_step=False,
                    N_iter=nt_pmp)
    solver.setup()
    solver.solve_fancy()

    t_end = time.time()
    time_pmp = t_end - t_start

    # Analytic optimal trajectory
    w = -gradient[0, 1]
    theta_f = 0.01

    def analytic_traj(theta, theta_f):
        x = 0.5 * v_a / w * (-1 / np.cos(theta_f) * (np.tan(theta_f) - np.tan(theta)) +
                             np.tan(theta) * (1 / np.cos(theta_f) - 1 / np.cos(theta)) -
                             np.log((np.tan(theta_f)
                                     + 1 / np.cos(theta_f)) / (
                                            np.tan(theta) + 1 / np.cos(theta))))
        y = v_a / w * (1 / np.cos(theta) - 1 / np.cos(theta_f))
        return x + x_f, y

    def residual(theta_f):
        return analytic_traj(-theta_f, theta_f)[0]

    theta_f = scipy.optimize.newton_krylov(residual, 0.01)
    # print(f'theta_f : {theta_f}')

    points = np.array(list(map(lambda theta: analytic_traj(theta, theta_f), np.linspace(-theta_f, theta_f, 1000))))
    alyt_traj = Trajectory(np.zeros(points.shape[0]),
                           points,
                           np.zeros(points.shape[0]),
                           points.shape[0] - 1,
                           coords=COORD_CARTESIAN)
    mp.trajs.append(alyt_traj)
    mdfm.dump_trajs(mp.trajs)

    params = {
        'coords': 'cartesian',
        'bl_wind': (total_wind.grid[0, 0, 0], total_wind.grid[0, 0, 1]),
        'tr_wind': (total_wind.grid[-1, -1, 0], total_wind.grid[-1, -1, 1]),
        'nx_wind': total_wind.grid.shape[0],
        'ny_wind': total_wind.grid.shape[1],
        'date_wind': total_wind.ts[0],
        'point_init': (x_init[0], x_init[1]),
        'max_time': T,
        'nt_pmp': nt_pmp,
        'pmp_time': time_pmp
    }

    ps = ParamsSummary(params, output_dir)
    ps.dump()


if __name__ == '__main__':
    example_linear_wind()
