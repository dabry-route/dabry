import os
import time
from math import atan2

from datetime import datetime, timedelta
import numpy as np
import scipy.optimize

from dabry.feedback import FunFB, ConstantFB, GSTargetFB, GreatCircleFB, HTargetFB
from dabry.mdf_manager import DDFmanager
from dabry.params_summary import ParamsSummary
from dabry.misc import Chrono
from dabry.problem import IndexedProblem, DatabaseProblem
from dabry.shooting import Shooting
from dabry.solver_ef import SolverEF
from dabry.solver_rp import SolverRP
from dabry.stoppingcond import TimedSC, DistanceSC
from dabry.wind import DiscreteWind

if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 0, 0
    dbpb = None#'37W_8S_16W_17S_20220301_12'
    cache_rff = True
    cache_wind = True

    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = DDFmanager()
    if dbpb is not None:
        output_dir = f'/home/bastien/Documents/work/mermoz/output/example_solver-ef_{dbpb}'
    else:
        output_dir = f'/home/bastien/Documents/work/mermoz/output/example_solver-ef_{IndexedProblem.problems[pb_id][1]}'
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir(keep_rff=cache_rff, keep_wind=cache_wind)

    nx_rft = 101
    ny_rft = 101
    nt_rft = 20

    # Create problem
    # mdfm.dump_wind_from_grib2(grib_fps, bl, tr)
    # pb = DatabaseProblem('/home/bastien/Documents/data/wind/ncdc/tmp.mz/wind.h5')
    if dbpb is not None:
        pb = DatabaseProblem(os.path.join('/home/bastien/Documents/data/wind/ncdc/', dbpb, 'wind.h5'), airspeed=23.)
    else:
        pb = IndexedProblem(pb_id, seed=seed)

    if not cache_wind:
        chrono.start('Dumping windfield to file')
        mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
        chrono.stop()

    # Setting the solver
    t_upper_bound = pb.time_scale if pb.time_scale is not None else 1.2 * pb.geod_l / pb.model.v_a
    solver_ef = solver = SolverEF(pb, t_upper_bound, max_steps=500, rel_nb_ceil=0.05)

    chrono.start('Computing EF')
    res = solver.solve()
    reach_time, iit, p_init = res.duration, res.index, res.adjoint
    mdfm.dump_trajs([res.traj])
    chrono.stop()
    print(f'Reach time : {reach_time / 3600:.2f}')

    trajs = solver.get_trajs()
    m = None
    k0 = -1
    for k, traj in enumerate(trajs):
        candidate = np.min(np.linalg.norm(traj.points - pb.x_target, axis=1))
        if m is None or m > candidate:
            m = candidate
            k0 = k

    solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft)
    if cache_rff:
        solver_rp.rft.load_cache(os.path.join(output_dir, 'rff.h5'))
    solver_rp.solve()

    if not cache_rff:
        solver_rp.rft.dump_rff(output_dir)

    pb.load_feedback(FunFB(lambda x: solver_rp.control(x, backward=True), no_time=True))
    sc = DistanceSC(lambda x: pb.distance(x, pb.x_init), pb.geod_l * 0.001)
    # sc = TimedSC(pb.model.wind.t_start)
    traj = pb.integrate_trajectory(pb.x_target, sc, int_step=reach_time / iit,
                                   t_init=pb.model.wind.t_start + reach_time,
                                   backward=True)
    traj.flip()
    mdfm.dump_trajs([traj])

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
    """
    chrono.start('Optimal trajectory Extremals')
    pb.load_feedback(FunFB(solver.control))
    sc = TimedSC(pb.model.wind.t_start + solver.reach_time)
    traj = pb.integrate_trajectory(pb.x_init, sc, int_step=reach_time / iit, t_init=pb.model.wind.t_start)
    traj.type = TRAJ_OPTIMAL
    traj.info = 'Extremals'
    traj = pb.integrate_trajectory(pb.x_init, sc, int_step=reach_time / iit / 10, t_init=pb.model.wind.t_start)
    traj.type = TRAJ_OPTIMAL
    traj.info = 'Extremals10'
    chrono.stop()

    if not cache_rff:
        chrono.start('Computing RFFs')
        solver_rp = solver = SolverRP(pb, nx_rft, ny_rft, nt_rft, max_time=pb.model.wind.t_end - pb.model.wind.t_start)
        solver.solve()
        chrono.stop()
        solver_rp.rft.dump_rff(output_dir)

        chrono.start('Optimal trajectory RFF')
        tu = solver.rft.get_time(pb.x_target)
        pb.load_feedback(FunFB(lambda x: solver.control(x, backward=True), no_time=True))
        sc = DistanceSC(lambda x: pb.distance(x, pb.x_init), pb.geod_l * 0.01)
        #sc = TimedSC(pb.model.wind.t_start)
        traj = pb.integrate_trajectory(pb.x_target, sc, int_step=reach_time / iit,
                                       t_init=tu,
                                       backward=True)
        li = traj.last_index
        traj.timestamps[:li + 1] = traj.timestamps[:li + 1][::-1]
        traj.points[:li + 1] = traj.points[:li + 1][::-1]
        traj.type = TRAJ_OPTIMAL
        traj.info = 'RFT'
        traj = pb.integrate_trajectory(pb.x_target, sc, int_step=reach_time / iit,
                                       t_init=tu,
                                       backward=True)
        li = traj.last_index
        traj.timestamps[:li + 1] = traj.timestamps[:li + 1][::-1]
        traj.points[:li + 1] = traj.points[:li + 1][::-1]
        traj.type = TRAJ_OPTIMAL
        traj.info = 'RFT10'

        chrono.stop()

    chrono.start('GSTarget FB')
    pb.load_feedback(GSTargetFB(pb.model.wind, pb.model.v_a, pb.x_target, pb.coords))
    sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), pb.geod_l / 100.)
    traj = pb.integrate_trajectory(pb.x_init, sc, max_iter=2 * iit, int_step=reach_time / iit,
                                   t_init=pb.model.wind.t_start)
    traj.info = 'GSTarget FB'
    chrono.stop()

    chrono.start('HTarget FB')
    pb.load_feedback(HTargetFB(pb.x_target, pb.coords))
    sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), pb.geod_l / 100.)
    traj = pb.integrate_trajectory(pb.x_init, sc, max_iter=2 * iit, int_step=reach_time / iit,
                                   t_init=pb.model.wind.t_start)
    traj.info = 'HTarget FB'
    chrono.stop()
    """
    # pb.load_feedback(GreatCircleFB(pb.model.wind, pb.model.v_a, pb.x_target))
    # sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), pb.geod_l / 100.)
    # traj = pb.integrate_trajectory(pb.x_init, sc, max_iter=2 * iit, int_step=reach_time / iit, t_init=t_init)
    # traj.info = 'Great circle'

    chrono.start('Dumping data')
    mdfm.dump_trajs(trajs)
    mdfm.dump_trajs(pb.trajs)
    chrono.stop()

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    ps.dump()

    print(f'Results saved to {output_dir}')
