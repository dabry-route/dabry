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
from mermoz.rft import RFT
from mermoz.shooting import Shooting
from mermoz.solver import Solver
from mermoz.trajectory import Trajectory
from mermoz.wind import LinearWind, DiscreteWind, UniformWind, RankineVortexWind

mpl.style.use('seaborn-notebook')


def run():
    """
    Example of extremals and RFF tracking on a linear wind example
    """
    output_dir = '/home/bastien/Documents/work/mermoz/output/example_ft_vor3'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)

    # UAV airspeed in m/s
    v_a = 23.

    coords = COORD_CARTESIAN

    # The time window upper bound in seconds
    T = 20 * 3600

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

    factor = 1e6
    factor_speed = 23.
    x_f = factor * 1.
    x_target = np.array([x_f, 0.])

    offset = np.array([0., 0.])
    omega1 = factor * (np.array([0.5, 0.8]) + offset)
    omega2 = factor * (np.array([0.8, 0.2]) + offset)
    omega3 = factor * (np.array([0.6, -0.5]) + offset)

    vortex1 = RankineVortexWind(omega1[0], omega1[1], factor * factor_speed * -1., factor * 1e-1)
    vortex2 = RankineVortexWind(omega2[0], omega2[1], factor * factor_speed * -0.8, factor * 1e-1)
    vortex3 = RankineVortexWind(omega3[0], omega3[1], factor * factor_speed * 0.8, factor * 1e-1)
    print(vortex1.max_speed())
    print(vortex2.max_speed())
    print(vortex3.max_speed())
    const_wind = UniformWind(np.array([0., 0.]))

    alty_wind = 3. * const_wind + vortex1 + vortex2 + vortex3
    total_wind = DiscreteWind()
    total_wind.load_from_wind(alty_wind, 101, 101, bl, tr, coords='cartesian')
    mdfm.dump_wind(total_wind)

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = factor * np.array([0., 0.])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, axes_equal=False)

    # Setting front tracking algorithm
    nx_rft = 101
    ny_rft = 101

    delta_x = (total_wind.x_max - total_wind.x_min) / (nx_rft - 1)
    delta_y = (total_wind.y_max - total_wind.y_min) / (ny_rft - 1)

    print(f"Tracking reachability front ({nx_rft}x{ny_rft}, dx={delta_x:.2E}, dy={delta_y:.2E})... ", end='')
    t_start = time.time()

    rft = RFT(bl, tr, T, nx_rft, ny_rft, mp, x_init, kernel='matlab', coords=coords)

    rft.compute()

    t_end = time.time()
    time_rft = t_end - t_start
    print(f"Done ({time_rft:.3f} s)")

    rft.dump_rff(output_dir)

    # Set a list of initial adjoint states for the shooting method
    nt_pmp = 100
    initial_headings = np.linspace(0.1, 2 * np.pi - 0.1, 30)
    list_p = list(map(lambda theta: -np.array([np.cos(theta), np.sin(theta)]), initial_headings))

    print(f"Shooting PMP trajectories ({len(list_p)})... ", end='')
    t_start = time.time()

    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined
    def domain(x):
        if x[0] < bl[0] or x[0] > tr[0] or x[1] < bl[1] or x[1] > tr[1]:
            return False
        return True

    for k, p in enumerate(list_p):
        shoot = Shooting(zermelo_model.dyn, x_init, T, adapt_ts=False, N_iter=nt_pmp, domain=domain, coords=coords)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)

    t_end = time.time()
    time_pmp = t_end - t_start
    print(f"Done ({time_pmp:.3f} s)")

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
        'nt_rft': rft.nt,
        'nx_rft': nx_rft,
        'ny_rft': ny_rft,
        'rft_time': time_rft,
        'pmp_time': time_pmp
    }

    ps = ParamsSummary(params, output_dir)
    ps.dump()


if __name__ == '__main__':
    run()
