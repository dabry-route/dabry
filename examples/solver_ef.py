import time
from math import atan2

from datetime import datetime, timedelta
import numpy as np
import scipy.optimize

from mermoz.feedback import FunFB, ConstantFB
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem
from mermoz.shooting import Shooting
from mermoz.solver_ef import SolverEF
from mermoz.solver_rp import SolverRP
from mermoz.stoppingcond import TimedSC
from mermoz.wind import DiscreteWind

if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 2, 0
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

    # _, nx_uv, ny_uv, _ = pb.model.wind.uv.shape
    # pb.model.wind.uv = 0.01 * np.ones((1, nx_uv, ny_uv, 2))

    # Log windfield to file
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)

    # Setting the solver
    solver = SolverEF(pb, 1.2 * pb._geod_l / pb.model.v_a, max_steps=400, hard_obstacles=not seed)
    # solver = SolverRP(pb, nx_rft, ny_rft, nt_rft, extremals=False)

    t_start = time.time()
    reach_time, iit, p_init = solver.solve()
    print(f'iit : {iit}')
    t_end = time.time()
    time_ef = t_end - t_start
    print(f'Dual EF solved in {time_ef:.3f}s')
    solver.set_primal(True)
    solver.solve()

    trajs = solver.get_trajs(dual_only=True)
    # m = None
    # k0 = -1
    # for k, traj in enumerate(trajs):
    #     candidate = np.min(np.linalg.norm(traj.points - pb.x_target, axis=1))
    #     if m is None or m > candidate:
    #         m = candidate
    #         k0 = k

    # solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft, extremals=False)
    # if cache:
    #     solver_rp.rft.load_cache(os.path.join(output_dir, 'rff.h5'))
    # solver_rp.solve()
    # if not cache:
    # solver.rft.dump_rff(output_dir)

    """
    x_dim = np.linalg.norm(pb.x_init - pb.x_target)
    def shoot(u0):
        shooting = Shooting(pb.model.dyn, pb.x_init, reach_time)
        shooting.set_adjoint(-np.array((np.cos(u0), np.sin(u0))).reshape((2,)))
        return shooting.integrate()


    def loss(u0):
        traj = shoot(u0)
        err = (traj.points[traj.last_index] - pb.x_target) / x_dim
        return err @ err


    u_p_init = np.pi + atan2(*p_init[::-1])
    traj = shoot(u_p_init)
    traj.type = TRAJ_INT
    pb.trajs.append(traj)

    sol = scipy.optimize.brute(loss, ((u_p_init-0.01, u_p_init+0.01),))
    print(sol)
    traj = shoot(sol)
    traj.type = 'optimal'
    pb.trajs.append(traj)

    u_list = np.linspace(u_p_init, (3 * u_p_init + sol) / 4., 10)
    for u in u_list:
        traj = shoot(u)
        traj.type = TRAJ_INT
        pb.trajs.append(traj)
    """

    pb.load_feedback(FunFB(solver.control))
    args = (pb.model.wind.t_start + solver.max_time,) + \
           (() if pb.model.wind.t_end is None else (pb.model.wind.t_end,))
    sc = TimedSC(*args)
    if pb.model.wind.t_end is None:
        t_init = pb.model.wind.t_start
    else:
        t_init = pb.model.wind.t_start + solver.max_time - reach_time
    print(timedelta(seconds=reach_time))

    traj = pb.integrate_trajectory(pb.x_init, sc, int_step=reach_time / iit, t_init=t_init)
    traj.type = TRAJ_OPTIMAL
    traj.info = 'Extremals'

    solver = SolverRP(pb, nx_rft, ny_rft, nt_rft, max_time=1.3 * pb._geod_l / pb.model.v_a)
    solver.solve()
    solver.rft.dump_rff(output_dir)
    mdfm.dump_trajs(solver.mp_dual.trajs)

    mdfm.dump_trajs(trajs)
    mdfm.dump_trajs(pb.trajs)

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    ps.dump()

    print(f'Results saved to {output_dir}')
