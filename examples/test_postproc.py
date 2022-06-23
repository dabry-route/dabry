import os
import time

import matplotlib as mpl
import numpy as np
import scipy.optimize

from mdisplay.geodata import GeoData

from mermoz.misc import *
from mermoz.params_summary import ParamsSummary
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.rft import RFT
from mermoz.shooting import Shooting
from mermoz.solver import Solver
from mermoz.solver_rp import SolverRP
from mermoz.trajectory import Trajectory
from mermoz.wind import DiscreteWind
from mermoz.mdf_manager import MDFmanager
from mermoz.post_processing import PostProcessing

mpl.style.use('seaborn-notebook')


def run():
    """
    Example of reachability front tracking
    """

    coords = COORD_CARTESIAN

    output_dir = '../output/example_test_postproc/'
    wind_data_dir = '/home/bastien/Documents/data/wind/ncdc/test-postproc.mz'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir()

    print("Example : Test postproc")
    print("Building model... ", end='')
    t_start = time.time()
    # UAV airspeed in m/s
    v_a = 23.
    # The time window upper bound in seconds

    # Load wind
    total_wind = DiscreteWind(force_analytical=True)
    total_wind.load(os.path.join(wind_data_dir, 'wind.h5'))
    mdfm.dump_wind(total_wind)

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
    zermelo_model.update_wind(total_wind)

    # Get problem domain boundaries
    bl = np.zeros(2)
    tr = np.zeros(2)
    bl[:] = total_wind.grid[1, 1]
    tr[:] = total_wind.grid[-2, -2]
    offset = (tr - bl) * 0.05

    # Initial point
    x_init = np.zeros(2)
    x_init[:] = bl + offset

    x_target = np.zeros(2)
    x_target[:] = tr - offset

    T = distance(x_init, x_target, coords=coords) / v_a * 1.2
    print(T/3600.)

    print(np.linalg.norm(x_target - x_init))

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, x_init, x_target, coords=coords)

    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(mp)
    ps.dump()

    pp = PostProcessing(output_dir)
    L = np.linalg.norm(x_target - x_init)
    cr = np.array(((0., -1.), (1., 0.))) @ (x_target - x_init)
    cr = cr / np.linalg.norm(cr)

    def fourier_traj(x):
        points = np.einsum('i,j->ij', 1 - s, x_init) + np.einsum('i,j->ij', s, x_target)
        for i, x in enumerate(x):
            points += np.einsum('i,j->ij', np.sin((i + 1) * pi * s), x * cr)
        return points

    def loss(x):
        return pp.point_stats(fourier_traj(x)).duration

    nt = 1000
    s = np.linspace(0., 1., nt)
    l0 = L / 10.

    traj = Trajectory(np.zeros(nt),
                      fourier_traj([0.]),
                      np.zeros(nt),
                      nt,
                      coords=coords)
    mp.trajs.append(traj)

    traj = Trajectory(np.zeros(nt),
                      fourier_traj([l0]),
                      np.zeros(nt),
                      nt,
                      coords=coords)
    mp.trajs.append(traj)

    traj = Trajectory(np.zeros(nt),
                      fourier_traj([l0, l0]),
                      np.zeros(nt),
                      nt,
                      coords=coords)
    mp.trajs.append(traj)

    traj = Trajectory(np.zeros(nt),
                      fourier_traj([-l0, l0/2, l0/3]),
                      np.zeros(nt),
                      nt,
                      coords=coords)
    mp.trajs.append(traj)


    l0 = L/10.
    # sol = scipy.optimize.brute(loss, ((0., l0), (0., l0)))
    # print(sol)
    # traj = Trajectory(np.zeros(nt),
    #                   fourier_traj(sol),
    #                   np.zeros(nt),
    #                   nt,
    #                   coords=coords)
    # mp.trajs.append(traj)

    # for N in [1, 2]:
    #     for ampl in [L/10., -L/10., L/20., -L/20.]:
    #         traj = Trajectory(np.zeros(nt),
    #                           np.einsum('i,j->ij', 1 - s, x_init) + np.einsum('i,j->ij', s, x_target) +
    #                           np.einsum('i,j->ij', np.sin(N * pi * s), ampl * cr),
    #                           np.zeros(nt),
    #                           nt,
    #                           coords=coords)
    #         mp.trajs.append(traj)



    nt_pmp = 1000
    opti_ceil = L * 0.005
    neighb_ceil = opti_ceil / 2.

    u_min = DEG_TO_RAD * 10.
    u_max = DEG_TO_RAD * 90.

    #solver = SolverRP(mp, 51, 51, 25)
    solver = Solver(mp,
                    x_init,
                    x_target,
                    T,
                    u_min,
                    u_max,
                    output_dir,
                    N_disc_init=10,
                    opti_ceil=opti_ceil,
                    neighb_ceil=neighb_ceil,
                    n_min_opti=1,
                    adaptive_int_step=False,
                    nt_pmp=nt_pmp)
    solver.solve_fancy(dump_only_opti=True)
    #solver.solve()

    mdfm.dump_trajs(mp.trajs)
    #mdfm.dump_trajs(solver.mp_pmp.trajs)
    #solver.rft.dump_rff(output_dir)

    factor = RAD_TO_DEG if coords == 'gcs' else 1.

    params = {
        'coords': coords,
        'bl_wind': (factor * total_wind.grid[0, 0, 0], factor * total_wind.grid[0, 0, 1]),
        'tr_wind': (factor * total_wind.grid[-1, -1, 0], factor * total_wind.grid[-1, -1, 1]),
        'nx_wind': total_wind.grid.shape[0],
        'ny_wind': total_wind.grid.shape[1],
        'date_wind': total_wind.ts[0],
        'point_init': tuple(x_init),
        'point_target': tuple(x_target),
        'max_time': T,
        'va': v_a
    }

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_solver(solver)
    ps.dump()


if __name__ == '__main__':
    run()
