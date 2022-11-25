import os
import time
from math import atan2

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from tqdm import tqdm

from mermoz.feedback import FunFB, ConstantFB, GSTargetFB, GreatCircleFB, HTargetFB, FixedHeadingFB, FunAS, E_GSTargetFB
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem, DatabaseProblem
from mermoz.shooting import Shooting
from mermoz.solver_ef import SolverEF
from mermoz.trajectory import Trajectory
from mermoz.solver_rp import SolverRP
from mermoz.stoppingcond import TimedSC, DistanceSC, DisjunctionSC
from mermoz.wind import DiscreteWind

if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 11, 1
    dbpb = None  # '37W_8S_16W_17S_20220301_12'
    cache_rff = False
    cache_wind = False

    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    if dbpb is not None:
        output_dir = f'/home/bastien/Documents/work/mermoz/output/example_energy_{dbpb}'
    else:
        output_dir = f'/home/bastien/Documents/work/mermoz/output/example_energy_{IndexedProblem.problems[pb_id][1]}'
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

    # Sequence
    """
    pb.update_airspeed(pb.aero.v_minp)
    asp_offlist = [0., 2., 4., 6., 8.]
    for asp_offset in asp_offlist:
        chrono.start('Computing Energy EF')
        t_upper_bound = 2 * pb._geod_l / pb.aero.v_minp
        solver_ef = solver = SolverEF(pb, t_upper_bound, mode=1, max_steps=300, rel_nb_ceil=0.01, cost_ceil=30*3.6e6,
                                    asp_offset=asp_offset)
        reach_time, iit, p_init = solver.solve(forward_only=True)
        reach_time_tef = reach_time
        chrono.stop()
        print(f'Reach time : {reach_time / 3600:.2f}')
        # trajs = solver.get_trajs()
        # mdfm.dump_trajs(trajs)

        chrono.start('Computing optimal traj')
        traj = solver_ef.build_opti_traj(force_primal=True)
        traj.info = traj.info + f'_v+{int(asp_offset)}'
        if traj.timestamps.shape[0] > 0:
            mdfm.dump_trajs([traj])
        chrono.stop()
    
    for asp_offset in asp_offlist:
        chrono.start('Computing Time EF')
        pb.update_airspeed(pb.aero.v_minp + asp_offset)
        t_upper_bound = 2 * pb._geod_l / pb.aero.v_minp
        solver_ef = solver = SolverEF(pb, t_upper_bound, mode=0, max_steps=300, rel_nb_ceil=0.01, cost_ceil=30*3.6e6)
        reach_time, iit, p_init = solver.solve(forward_only=True)
        reach_time_tef = reach_time
        chrono.stop()
        print(f'Reach time : {reach_time / 3600:.2f}')
        # trajs = solver.get_trajs()
        # mdfm.dump_trajs(trajs)

        chrono.start('Computing optimal traj')
        traj = solver_ef.build_opti_traj(force_primal=True)
        traj.info = traj.info + f'_v+{int(asp_offset)}'
        if traj.timestamps.shape[0] > 0:
            mdfm.dump_trajs([traj])
        chrono.stop()
    """
    """
    chrono.start('Computing Min Energy EF')
    t_upper_bound = 2 * pb._geod_l / pb.aero.v_minp
    solver_ef = solver = SolverEF(pb, t_upper_bound, mode=1, max_steps=300, rel_nb_ceil=0.01, cost_ceil=60 * 3.6e6, asp_offset=20.)
    reach_time, iit, p_init = solver.solve(forward_only=True)
    reach_time_tef = reach_time
    chrono.stop()
    print(f'Reach time : {reach_time / 3600:.2f}')
    trajs = solver.get_trajs()
    mdfm.dump_trajs(trajs)

    chrono.start('Computing optimal traj')
    traj = solver_ef.build_opti_traj(force_primal=True)
    traj.info = traj.info
    if traj.timestamps.shape[0] > 0:
        mdfm.dump_trajs([traj])
    chrono.stop()
    """

    chrono.start('Computing Time EF')
    t_upper_bound = 2 * pb._geod_l / pb.model.v_a
    solver_ef = solver = SolverEF(pb, t_upper_bound, mode=0, max_steps=100, rel_nb_ceil=0.01, dt=0.01*pb._geod_l / pb.model.v_a)
    reach_time, iit, p_init = solver.solve(forward_only=True)
    reach_time_tef = reach_time
    chrono.stop()
    print(f'Reach time : {reach_time / 3600:.2f}')
    trajs = solver.get_trajs()
    mdfm.dump_trajs(trajs)

    chrono.start('Computing optimal traj')
    traj = solver_ef.build_opti_traj(force_primal=True)
    if traj.timestamps.shape[0] > 0:
        mdfm.dump_trajs([traj])
    chrono.stop()

    # chrono.start('Computing Energy EF')
    # t_upper_bound = 3 * pb._geod_l / pb.aero.v_minp
    # solver_ef = solver = SolverEF(pb, t_upper_bound, mode=1, max_steps=500, rel_nb_ceil=0.025)
    # reach_time, iit, p_init = solver.solve(forward_only=True)
    # chrono.stop()
    # print(f'Reach time : {reach_time / 3600:.2f}')
    # trajs = solver.get_trajs()
    # mdfm.dump_trajs(trajs)
    #
    # chrono.start('Computing optimal traj')
    # traj = solver_ef.build_opti_traj(force_primal=True)
    # mdfm.dump_trajs([traj])
    # chrono.stop()


    def f(a, pn, rtraj=False):
        angle = 2 * np.arctan(a)  # maps a to -pi, pi
        s = np.array((np.cos(angle), np.sin(angle)))
        shooting = Shooting(pb.model.dyn, pb.x_init, reach_time_tef, mode='energy-opt', domain=pb.domain)
        shooting.set_adjoint((-1. * pn * s).reshape((2,)))
        traj = shooting.integrate()
        dist = np.array(list(map(lambda x: pb.distance(x, pb.x_target), traj.points)))
        if rtraj:
            return traj
        return np.min(dist)

    """
    for pn0 in [10, 20, 30, 40]:
        chrono.start(f'Minimizing for {pn0}')
        res = scipy.optimize.minimize(lambda a: f(a, pn0), np.zeros(1))
        traj = f(res.x, pn0, rtraj=True)
        traj.info = f'trv_{int(pn0)}_grad'
        trajs.append(traj)

    mdfm.dump_trajs(trajs)
    """
        # err = np.min(np.linalg.norm(traj.points - pb.x_target, axis=1) / pb._geod_l)
        # reached = err < 0.02
        # if reached:
        #     print('|', end='')
        #     candidates.append((pn0, psi / np.pi))
        #     trajs.append(traj)

    """
    chrono.start('Computing optimal traj')
    traj = solver_ef.build_opti_traj(force_dual=True)
    mdfm.dump_trajs([traj])
    chrono.stop()
    """
    """
    for factor in [2, 4, 6, 8, 10]:
        nrj_ceil = factor * 3.6e6
        virt_points = []
        pi_args = []
        for i, traj in enumerate(trajs):
            k0 = 0
            for k, nrj in enumerate(traj.costs):
                if nrj > nrj_ceil:
                    k0 = k
                    break
            if k0 > 0:
                virt_points.append(traj.points[k0])
                pi_args.append(atan2(*(-solver.p_inits[i])[::-1]))

        virt_points = [x for _, x in sorted(zip(pi_args, virt_points), key=lambda pair: pair[0])]

        if len(virt_points) == 0:
            continue

        nvp = len(virt_points)
        mdfm.dump_trajs([Trajectory(np.zeros(nvp), np.array(virt_points), np.zeros(nvp), nvp - 1, solver.mp.coords,
                                    info=f'{factor}kWh')])
    """
    """
    for v_a in np.linspace(20, 40, 5):
        pb.update_airspeed(v_a)
        t_upper_bound = 2 * pb._geod_l / pb.model.v_a
        solver_ef = solver = SolverEF(pb, t_upper_bound, mode=0, max_steps=500, rel_nb_ceil=0.05)
        reach_time, iit, p_init = solver.solve(forward_only=True)
        chrono.stop()
        print(f'Reach time : {reach_time / 3600:.2f}')
        trajs = solver.get_trajs()
        # mdfm.dump_trajs(trajs)

        if iit >= 0:
            chrono.start('Computing optimal traj')
            traj = solver_ef.build_opti_traj()
            traj.info = f'TEF-{v_a:.2f}'
            mdfm.dump_trajs([traj])
            chrono.stop()

    pb.load_multifb(E_GSTargetFB(pb.model.wind, pb.aero, pb.x_target, pb.coords))
    sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), 0.02 * pb._geod_l)
    traj = pb.integrate_trajectory(pb.x_init, sc, int_step=reach_time / iit, t_init=pb.model.wind.t_start)
    traj.type = TRAJ_INT
    traj.info = f'Target'
    mdfm.dump_trajs([traj])
    """
    """
    for factor in [2, 4, 6, 8, 10]:
        nrj_ceil = factor * 3.6e6
        virt_points = []
        pi_args = []
        for i, traj in enumerate(trajs):
            k0 = 0
            for k, nrj in enumerate(traj.costs):
                if nrj > nrj_ceil:
                    k0 = k
                    break
            if k0 > 0:
                virt_points.append(traj.points[k0])
                pi_args.append(atan2(*(-solver.p_inits[i])[::-1]))

        virt_points = [x for _, x in sorted(zip(pi_args, virt_points), key=lambda pair: pair[0])]

        if len(virt_points) == 0:
            continue

        nvp = len(virt_points)
        mdfm.dump_trajs([Trajectory(np.zeros(nvp), np.array(virt_points), np.zeros(nvp), nvp - 1, solver.mp.coords,
                                    info=f'{factor}kWh (CTAS)')])
    """
    # chrono.start('Optimal trajectory Extremals')
    # pb.load_feedback(FunFB(solver.control))
    # pb.load_aslaw(FunAS(solver.aslaw))
    # sc = TimedSC(pb.model.wind.t_start + solver.reach_time)
    # traj = pb.integrate_trajectory(pb.x_init, sc, int_step=reach_time / iit, t_init=pb.model.wind.t_start)
    # traj.type = TRAJ_OPTIMAL
    # traj.info = f'Extremals_{v_a:.2f}'
    # chrono.stop()
    # mdfm.dump_trajs([traj])

    # for v_a in np.linspace(12., 25., 5):
    #     pb.update_airspeed(v_a)
    #     t_upper_bound = pb.time_scale if pb.time_scale is not None else 1.2 * pb._geod_l / pb.model.v_a
    #     chrono.start('Computing EF')
    #     solver_ef = solver = SolverEF(pb, t_upper_bound, max_steps=500, rel_nb_ceil=0.05)
    #     reach_time, iit, p_init = solver.solve(no_fast=True)
    #     chrono.stop()
    #     print(f'Reach time : {reach_time / 3600:.2f}')
    #     trajs = solver.get_trajs()
    #     mdfm.dump_trajs(trajs)
    #
    #     chrono.start('Optimal trajectory Extremals')
    #     pb.load_feedback(FunFB(solver.control))
    #     sc = TimedSC(pb.model.wind.t_start + solver.reach_time)
    #     traj = pb.integrate_trajectory(pb.x_init, sc, int_step=reach_time / iit, t_init=pb.model.wind.t_start)
    #     traj.type = TRAJ_OPTIMAL
    #     traj.info = 'Extremals'
    #     chrono.stop()
    # m = None
    # k0 = -1
    # for k, traj in enumerate(trajs):
    #     candidate = np.min(np.linalg.norm(traj.points - pb.x_target, axis=1))
    #     if m is None or m > candidate:
    #         m = candidate
    #         k0 = k

    # solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft, extremals=False)
    # if cache_rff:
    #     solver_rp.rft.load_cache(os.path.join(output_dir, 'rff.h5'))
    # solver_rp.solve()
    # if not cache_rff:
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

    #
    # if not cache_rff:
    #     chrono.start('Computing RFFs')
    #     solver_rp = solver = SolverRP(pb, nx_rft, ny_rft, nt_rft, max_time=pb.model.wind.t_end - pb.model.wind.t_start)
    #     solver.solve()
    #     chrono.stop()
    #     solver_rp.rft.dump_rff(output_dir)
    #
    #     chrono.start('Optimal trajectory RFF')
    #     tu = solver.rft.get_time(pb.x_target)
    #     pb.load_feedback(FunFB(lambda x: solver.control(x, backward=True), no_time=True))
    #     sc = DistanceSC(lambda x: pb.distance(x, pb.x_init), pb._geod_l * 0.01)
    #     # sc = TimedSC(pb.model.wind.t_start)
    #     traj = pb.integrate_trajectory(pb.x_target, sc, int_step=reach_time / iit,
    #                                    t_init=tu,
    #                                    backward=True)
    #     li = traj.last_index
    #     traj.timestamps[:li + 1] = traj.timestamps[:li + 1][::-1]
    #     traj.points[:li + 1] = traj.points[:li + 1][::-1]
    #     traj.type = TRAJ_OPTIMAL
    #     traj.info = 'RFT'
    #     traj = pb.integrate_trajectory(pb.x_target, sc, int_step=reach_time / iit,
    #                                    t_init=tu,
    #                                    backward=True)
    #     li = traj.last_index
    #     traj.timestamps[:li + 1] = traj.timestamps[:li + 1][::-1]
    #     traj.points[:li + 1] = traj.points[:li + 1][::-1]
    #     traj.type = TRAJ_OPTIMAL
    #     traj.info = 'RFT10'
    #
    #     chrono.stop()
    #
    # chrono.start('GSTarget FB')
    # pb.load_feedback(GSTargetFB(pb.model.wind, pb.model.v_a, pb.x_target, pb.coords))
    # sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), pb._geod_l / 100.)
    # traj = pb.integrate_trajectory(pb.x_init, sc, max_iter=2 * iit, int_step=reach_time / iit,
    #                                t_init=pb.model.wind.t_start)
    # traj.info = 'GSTarget FB'
    # chrono.stop()
    #
    # chrono.start('HTarget FB')
    # pb.load_feedback(HTargetFB(pb.x_target, pb.coords))
    # sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), pb._geod_l / 100.)
    # traj = pb.integrate_trajectory(pb.x_init, sc, max_iter=2 * iit, int_step=reach_time / iit,
    #                                t_init=pb.model.wind.t_start)
    # traj.info = 'HTarget FB'
    # chrono.stop()

    # pb.load_feedback(GreatCircleFB(pb.model.wind, pb.model.v_a, pb.x_target))
    # sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), pb._geod_l / 100.)
    # traj = pb.integrate_trajectory(pb.x_init, sc, max_iter=2 * iit, int_step=reach_time / iit, t_init=t_init)
    # traj.info = 'Great circle'

    # chrono.start('Dumping data')
    #
    # mdfm.dump_trajs(pb.trajs)
    # chrono.stop()

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    ps.dump()

    print(f'Results saved to {output_dir}')
