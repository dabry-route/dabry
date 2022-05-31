import os
import time

import matplotlib as mpl
import numpy as np

from mdisplay.geodata import GeoData

from mermoz.misc import *
from mermoz.params_summary import ParamsSummary
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.rft import RFT
from mermoz.shooting import Shooting
from mermoz.wind import DiscreteWind
from mermoz.mdf_manager import MDFmanager

mpl.style.use('seaborn-notebook')


def run():
    """
    Example of reachability front tracking
    """

    coords = COORD_CARTESIAN

    output_dir = '../output/example_test_flatten/'
    wind_data_dir = '/home/bastien/Documents/data/wind/ncdc/test-flattened-ortho.mz'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)

    print("Example : Test flatten")
    print("Building model... ", end='')
    t_start = time.time()
    # UAV airspeed in m/s
    v_a = 23.
    # The time window upper bound in seconds
    T = 20 * 3600.

    # Load wind
    total_wind = DiscreteWind(force_analytical=True)
    total_wind.load(os.path.join(wind_data_dir, 'wind.h5'), unstructured=True)
    mdfm.dump_wind(total_wind)

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
    zermelo_model.update_wind(total_wind)

    # Get problem domain boundaries
    bl = np.zeros(2)
    tr = np.zeros(2)
    bl[:] = total_wind.grid[1, 1]
    tr[:] = total_wind.grid[-2, -2]

    # Initial point
    x_init = 0.5 * (bl + tr)

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map')

    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")

    # Setting front tracking algorithm
    nx_rft = 201
    ny_rft = 201

    delta_x = (total_wind.x_max - total_wind.x_min) / (nx_rft - 1)
    delta_y = (total_wind.y_max - total_wind.y_min) / (ny_rft - 1)

    print(f"Tracking reachability front ({nx_rft}x{ny_rft}, dx={delta_x:.2E}, dy={delta_y:.2E})... ", end='')
    t_start = time.time()

    rft = RFT(bl, tr, T, nx_rft, ny_rft, 10, mp, x_init, kernel='matlab', coords=coords)

    rft.compute()
    rft.dump_rff(output_dir)

    t_end = time.time()
    time_rft = t_end - t_start
    print(f"Done ({time_rft:.3f} s)")

    # Set a list of initial adjoint states for the shooting method
    nt_pmp = 3000
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

    factor = RAD_TO_DEG if coords == 'gcs' else 1.

    params = {
        'coords': coords,
        'bl_wind': (factor * total_wind.grid[0, 0, 0], factor * total_wind.grid[0, 0, 1]),
        'tr_wind': (factor * total_wind.grid[-1, -1, 0], factor * total_wind.grid[-1, -1, 1]),
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
