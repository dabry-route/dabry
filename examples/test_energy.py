import os
import time
from math import atan2

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from tqdm import tqdm

from mermoz.feedback import FunFB, ConstantFB, GSTargetFB, GreatCircleFB, HTargetFB, FixedHeadingFB
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem, DatabaseProblem
from mermoz.shooting import Shooting
from mermoz.solver_ef import SolverEF
from mermoz.solver_rp import SolverRP
from mermoz.stoppingcond import TimedSC, DistanceSC, DisjunctionSC
from mermoz.wind import DiscreteWind

if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 0, 0
    dbpb = None  # '37W_8S_16W_17S_20220301_12'
    cache_rff = False
    cache_wind = True

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

    # Setting the solver

    t_upper_bound = pb.time_scale if pb.time_scale is not None else 1.2 * pb._geod_l / pb.model.v_a
    trajs = []
    """
    for a in np.linspace(0.01 * np.pi + 0.01, 0.15 * np.pi - 0.01, 3):
        shooting = Shooting(pb.model.dyn, pb.x_init, t_upper_bound, mode='energy-opt', domain=pb.domain)
        shooting.set_adjoint(100. * -1. * np.array((np.cos(a), np.sin(a))))
        trajs.append(shooting.integrate())
        # shooting = Shooting(pb.model.dyn, pb.x_init, t_upper_bound, mode='time-opt', domain=pb.domain)
        # shooting.set_adjoint(100. * -1. * np.array((np.cos(a), np.sin(a))))
        # trajs.append(shooting.integrate())
    """
    """
    l = np.linspace(-0.2 * np.pi, 0.1 * np.pi, 10)
    # al, au = l[1], l[2]
    # l2 = np.linspace(al, au, 4)
    l2 = l
    pn0s = [1., 10., 100., 1000.]
    fig, ax = plt.subplots()
    for k, a in enumerate(l2):
        for pn0 in pn0s:
            s = np.array((np.cos(a), np.sin(a)))
            cost = 'dobrokhodov'
            # pn0 = scipy.optimize.brentq(lambda pn:
            #                             power(airspeed_opti(np.array((pn, 0.)), cost=cost), cost=cost) - pn * (
            #                                     airspeed_opti(np.array((pn, 0.)), cost=cost) + s @ pb.model.wind.value(0., pb.x_init)),
            #                             1., 300.)
            #print(f'pn0 : {pn0:.2f}')
            # pn0 = pn0s[k]
            shooting = Shooting(pb.model.dyn, pb.x_init, 2 * t_upper_bound, mode='energy-opt', domain=pb.domain,
                                energy_ceil=20. * 8.4 * 3.6e6)
            shooting.set_adjoint(-1. * pn0 * s)
            traj = shooting.integrate()
            traj.info = f'trv'
            trajs.append(traj)
            point = pn0 * (-1. * s)
            ax.scatter(point[0], point[1])
    
    ax.grid(True)
    ax.axis('equal')
    #plt.show()
    """


    def auto_upperb(pn):
        va = airspeed_opti_(pn)
        return 3 * pb._geod_l / va

    good = [(5, 0.0626662),
            (10, 0.0775),
            (25, 0.0908558),
            (50, 0.093),
            (75, 0.0942757)]

    for pn0, factor in good:
        theta = factor * np.pi
        s = np.array((np.cos(theta), np.sin(theta)))
        shooting = Shooting(pb.model.dyn, pb.x_init, auto_upperb(pn0), mode='energy-opt', domain=pb.domain,
                            energy_ceil=20. * 8.4 * 3.6e6)
        shooting.set_adjoint(-1. * pn0 * s)
        traj = shooting.integrate()
        traj.info = f'trv_{int(pn0)}'
        trajs.append(traj)

    def f(factor):
        s = np.array((np.cos(factor * np.pi), np.sin(factor * np.pi))).reshape((2,))
        shooting = Shooting(pb.model.dyn, pb.x_init, auto_upperb(pn0), mode='energy-opt', domain=pb.domain,
                            energy_ceil=20. * 8.4 * 3.6e6)
        shooting.set_adjoint(-1. * pn0 * s)
        traj = shooting.integrate()
        return np.min(np.linalg.norm(traj.points - pb.x_target) / pb._geod_l)

    candidates =[]

    for pn0 in [5.]:
        print(pn0, end='')
        for factor in np.linspace(0.18, 0.22, 0):
            theta = factor * np.pi
            fb = FixedHeadingFB(pb.model.wind, airspeed_opti_(pn0), theta, coords=pb.coords)
            psi = fb.value(0, pb.x_init)
            s = np.array((np.cos(psi), np.sin(psi)))
            shooting = Shooting(pb.model.dyn, pb.x_init, auto_upperb(pn0), mode='energy-opt', domain=pb.domain,
                                energy_ceil=20. * 8.4 * 3.6e6)
            shooting.set_adjoint(-1. * pn0 * s)
            traj = shooting.integrate()
            traj.info = f'trv_{int(pn0)}_auto'
            err = np.min(np.linalg.norm(traj.points - pb.x_target, axis=1) / pb._geod_l)
            reached = err < 0.05
            if reached:
                print('|', end='')
                candidates.append((pn0, psi/np.pi))
                trajs.append(traj)
        print()
    print(candidates)
    mdfm.dump_trajs(trajs)

    """
    chrono.start('Computing EF')
    solver_ef = solver = SolverEF(pb, t_upper_bound, max_steps=500, rel_nb_ceil=0.05)
    reach_time, iit, p_init = solver.solve(no_fast=True)
    chrono.stop()
    print(f'Reach time : {reach_time / 3600:.2f}')
    trajs = solver.get_trajs()
    mdfm.dump_trajs(trajs)

    chrono.start('Optimal trajectory Extremals')
    pb.load_feedback(FunFB(solver.control))
    sc = TimedSC(pb.model.wind.t_start + solver.reach_time)
    traj = pb.integrate_trajectory(pb.x_init, sc, int_step=reach_time / iit, t_init=pb.model.wind.t_start)
    traj.type = TRAJ_OPTIMAL
    traj.info = 'Extremals'
    chrono.stop()
    """
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

    chrono.start('Dumping data')

    mdfm.dump_trajs(pb.trajs)
    chrono.stop()

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    ps.dump()

    print(f'Results saved to {output_dir}')
