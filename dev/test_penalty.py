import time
import os

import numpy as np

from dabry.ddf_manager import DDFmanager
from dabry.params_summary import ParamsSummary
from dabry.misc import Utils
from dabry.problem import IndexedProblem
from dabry.shooting import Shooting
from dabry.solver_ef import SolverEF
from dabry.solver_rp import SolverRP
from dabry.wind import DiscreteWind


if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 0, 0
    cache = False

    # Create a file manager to dump problem data
    mdfm = DDFmanager()
    output_dir = f'/home/bastien/Documents/work/mermoz/output/example_solver-ef_{IndexedProblem.problems[pb_id][1]}'
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir(keep_rff=cache)

    nx_rft = 101
    ny_rft = 101
    nt_rft = 20

    # Create problem
    pb = IndexedProblem(pb_id, seed=seed)
    # d_wind = DiscreteWind(interp='pwc')
    # d_wind.load_from_wind(pb.model.wind, nx_rft, ny_rft, pb.bl, pb.tr, pb.coords)
    # pb.model.wind = d_wind

    # Log windfield to file
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, bl=pb.bl, tr=pb.tr, nt=nt_rft)

    # Setting the solver
    solver = SolverEF(pb, max_steps=300, hard_obstacles=not seed, rel_nb_ceil=0.025)
    #solver = SolverRP(pb, nx_rft, ny_rft, nt_rft, extremals=False)

    t_start = time.time()
    reach_time, iit_opt, p_init = solver.solve()
    t_end = time.time()
    time_rp = t_end - t_start

    trajs = solver.get_trajs()
    m = None
    k0 = -1
    for k, traj in enumerate(trajs):
        candidate = np.min(np.linalg.norm(traj.points - pb.x_target, axis=1))
        if m is None or m > candidate:
            m = candidate
            k0 = k
    reach_index = np.argmin(np.linalg.norm(trajs[k0].points - pb.x_target, axis=1))
    reach_time = trajs[k0].timestamps[reach_index]

    # solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft, extremals=False)
    # if cache:
    #     solver_rp.rft.load_cache(os.path.join(output_dir, 'rff.h5'))
    # solver_rp.solve()
    # if not cache:
    #solver.rft.dump_rff(output_dir)
    shooting = Shooting(pb.model.dyn, pb.x_init, reach_time, N_iter=iit_opt)
    shooting.set_adjoint(p_init)
    traj = shooting.integrate()
    traj.type = 'optimal'
    pb.trajs.append(traj)

    mdfm.dump_trajs(trajs)
    mdfm.dump_trajs(pb.trajs)

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    ps.dump()

    print(f'Results saved to {output_dir}')
