import time
import os

import numpy as np

from mermoz.feedback import ConstantFB, TargetFB, FixedHeadingFB
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem
from mermoz.shooting import Shooting
from mermoz.solver_ef import SolverEF
from mermoz.solver_nlp import SolverNLP
from mermoz.solver_rp import SolverRP
from mermoz.stoppingcond import TimedSC
from mermoz.trajectory import Trajectory
from mermoz.wind import DiscreteWind

if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 0, 0
    cache = False

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    output_dir = f'/home/bastien/Documents/work/mermoz/output/example_solver-nlp_{IndexedProblem.problems[pb_id][1]}'
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

    solver_ef = SolverEF(pb, max_steps=200)

    reach_time, _, _ = solver_ef.solve()
    trajs = solver_ef.get_trajs()

    # Setting the solver
    n_steps = 50
    #pb.load_feedback(TargetFB(pb.model.wind, pb.model.v_a, pb.x_target, pb.coords))
    pb.load_feedback(FixedHeadingFB(pb.model.wind, pb.model.v_a, 0.55, pb.coords))
    #pb.load_feedback(ConstantFB(0.15))
    sc = TimedSC(0.95*reach_time)
    pb.integrate_trajectory(pb.x_init, sc, int_step=0.95 * reach_time / n_steps, max_iter=n_steps)
    # i_guess = np.zeros(2 * n_steps - 2)
    #
    # i_guess[:] = np.concatenate((
    #     np.array(list(map(lambda u: np.cos(u), pb.trajs[0].controls[:-1]))),
    #     np.array(list(map(lambda u: np.sin(u), pb.trajs[0].controls[:-1])))
    # ))

    i_guess = np.zeros(n_steps - 1)
    i_guess[:] = pb.trajs[0].controls[:-1]

    solver = SolverNLP(pb, 0.95 * reach_time, n_steps, i_guess=i_guess)
    # solver = SolverRP(pb, nx_rft, ny_rft, nt_rft, extremals=False)

    t_start = time.time()
    res, x_opt = solver.solve()
    t_end = time.time()
    time_rp = t_end - t_start

    print(res)

    pb.trajs.append(Trajectory(np.zeros(n_steps), x_opt, np.zeros(n_steps), n_steps, type=TRAJ_PATH, coords=pb.coords))

    mdfm.dump_trajs(trajs)
    mdfm.dump_trajs(pb.trajs)

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    ps.dump()

    print(f'Results saved to {output_dir}')
