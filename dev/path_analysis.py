import random
import os
import numpy as np
from numpy import pi
import sys
import time
import h5py

from dabry.problem import IndexedProblem
from dabry.misc import Utils
from dabry.trajectory import Trajectory
from dabry.post_processing import PostProcessing
from dabry.ddf_manager import DDFmanager
from dabry.params_summary import ParamsSummary
from dabry.solver_rp import SolverRP

if __name__ == '__main__':
    # Choose problem ID
    pb_id = 9

    output_dir = f'/home/bastien/Documents/work/mermoz/output/example_pa_{IndexedProblem.problems[pb_id][1]}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Create a file manager to dump problem data
    mdfm = DDFmanager()
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir()
    wind_fpath = '/home/bastien/Documents/data/wind/ncdc/test-postproc.mz/wind.h5'
    with h5py.File(wind_fpath, 'r') as f:
        bl = np.array((np.array(f['grid'][:, :, 0]).min(), np.array(f['grid'][:, :, 1]).min()))
        tr = np.array((np.array(f['grid'][:, :, 0]).max(), np.array(f['grid'][:, :, 1]).max()))

    # pb = DatabaseProblem(wind_fpath)
    pb = IndexedProblem(pb_id)

    if pb.coords == Utils.COORD_GCS:
        print('Not handling GCS for now', file=sys.stderr)
        exit(1)

    t_start = time.time()
    mdfm.dump_wind(pb.model.wind, nx=51, ny=51, bl=pb.bl, tr=pb.tr)
    t_end = time.time()
    print(t_end - t_start)

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    # ps.add_param('max_time', pb.geod_l / pb.model.v_a)

    L = np.linalg.norm(pb.x_target - pb.x_init)
    cr = np.array(((0., -1.), (1., 0.))) @ (pb.x_target - pb.x_init)
    cr = cr / np.linalg.norm(cr)

    ps.dump()
    pp = PostProcessing(output_dir)


    def fourier_traj(x, x_init, x_target, nt=1000):
        s = np.linspace(0., 1., nt)
        cr = np.array(((0., -1.), (1., 0.))) @ (pb.x_target - pb.x_init)
        cr = cr / np.linalg.norm(cr)
        points = np.einsum('i,j->ij', 1 - s, x_init) + np.einsum('i,j->ij', s, x_target)
        for i, x in enumerate(x):
            points += np.einsum('i,j->ij', np.sin((i + 1) * pi * s), x * cr)
        return points


    def loss(x, x_init, x_target):
        return pp.point_stats(fourier_traj(x, x_init, x_target)).duration


    nt = 1000
    s = np.linspace(0., 1., nt)
    l0 = L / 10.

    trajs = []

    # l0 = L / 10.
    import scipy.optimize

    # sol = scipy.optimize.minimize(loss, np.array((l0, l0)), bounds=((-2 * l0, 2 * l0), (-2 * l0, 2 * l0)),
    #                               method='L-BFGS-B', args=(pb.x_init, pb.x_target))
    # print(sol)
    # print(sol['x'])
    # traj = Trajectory(np.zeros(nt),
    #                   fourier_traj(sol['x'], pb.x_init, pb.x_target),
    #                   np.zeros(nt),
    #                   nt,
    #                   coords=pb.coords)
    # trajs.append(traj)
    i = 0
    N = 3
    N_samples = [5, 7, 9]
    for i_N in range(N):
        for _ in range(N_samples[i_N]):
            ampl = [- l0 + 2 * l0 * random.random() for _ in range(i_N)]
            traj = Trajectory(np.zeros(nt),
                              fourier_traj(ampl, pb.x_init, pb.x_target),
                              np.zeros(nt),
                              nt,
                              coords=pb.coords,
                              type=Utils.TRAJ_PATH,
                              label=i)
            i += 1
            trajs.append(traj)

    if pb_id == 1:
        # Linear wind case comes with an analytical solution
        alyt_traj = Utils.linear_wind_alyt_traj(pb.model.v_a, pb.model.wind.gradient[0, 1], pb.x_init, pb.x_target)
        trajs.append(alyt_traj)

    nx_rft = 51
    ny_rft = 51
    nt_rft = 10
    solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft, nt_pmp=1000, only_opti_extremals=True)

    t_start = time.time()
    solver_rp.solve()
    t_end = time.time()
    time_rp = t_end - t_start

    trajs.append(solver_rp.rft.backward_traj(pb.x_target, pb.x_init, solver_rp.opti_ceil,
                                             solver_rp.T, pb.model, N_disc=1000))


    mdfm.dump_trajs(trajs)
    mdfm.dump_trajs(solver_rp.mp_pmp.trajs)

    pp.stats()
