import time
import os

import numpy as np

from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem
from mermoz.shooting import Shooting
from mermoz.solver_ef import SolverEF
from mermoz.solver_rp import SolverRP
from mermoz.wind import DiscreteWind


if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 10, 1
    cache = False

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
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
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, bl=pb.bl, tr=pb.tr)

    # Setting the solver
    solver = SolverEF(pb, max_steps=120, hard_obstacles=not seed)
    #solver = SolverRP(pb, nx_rft, ny_rft, nt_rft, extremals=False)

    t_start = time.time()
    solver.solve()
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

    # solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft, extremals=False)
    # if cache:
    #     solver_rp.rft.load_cache(os.path.join(output_dir, 'rff.h5'))
    # solver_rp.solve()
    # if not cache:
    #solver.rft.dump_rff(output_dir)
    shooting = Shooting(pb.model.dyn, pb.x_init, pb._geod_l / pb.model.v_a)
    shooting.set_adjoint(solver.p_inits[k0])
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
