import os
import time
from math import atan2

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from tqdm import tqdm

from dabry.feedback import FunFB, ConstantFB, GSTargetFB, GreatCircleFB, HTargetFB, FixedHeadingFB
from dabry.ddf_manager import DDFmanager
from dabry.params_summary import ParamsSummary
from dabry.misc import Utils, Chrono
from dabry.problem import IndexedProblem, DatabaseProblem
from dabry.shooting import Shooting
from dabry.solver_ef import SolverEF
from dabry.solver_rp import SolverRP
from dabry.stoppingcond import TimedSC, DistanceSC, DisjunctionSC
from dabry.wind import DiscreteWind

if __name__ == '__main__':

    pb_params = '-35.16 -7.71 -14.24 11.52 202111011200 23.0 500'
    x_init, x_target, start_date, airspeed, level = Utils.read_pb_params(pb_params)

    duration = 2 * Utils.distance(Utils.DEG_TO_RAD * x_init, Utils.DEG_TO_RAD * x_target,
                                  coords=Utils.COORD_GCS) / airspeed
    stop_date = datetime.fromtimestamp(start_date.timestamp() + duration)

    ddf = DDFmanager()
    ddf.setup()

    ddf.retrieve_wind(start_date, stop_date, level=level, res='0.5')
    case_name = ddf.format_cname(x_init, x_target, start_date.timestamp())

    cache_wind = False
    cache_rff = False

    # This instance prints absolute elapsed time between operations
    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = DDFmanager(cache_wind=cache_wind, cache_rff=cache_rff)
    mdfm.setup()
    case_name = f'zermelo_{case_name}'
    mdfm.set_case(case_name)
    mdfm.clean_output_dir()

    # Space and time discretization
    # Will be used to save wind when wind is analytical and shall be sampled
    # Will also be used by front tracking module
    nx_rft = 101
    ny_rft = 101
    nt_rft = 20

    pb = DatabaseProblem(x_init=Utils.DEG_TO_RAD * x_init,
                         x_target=Utils.DEG_TO_RAD * x_target, airspeed=airspeed, t_start=start_date.timestamp(),
                         t_end=stop_date.timestamp(), altitude=level, resolution='0.5')

    # pb.flatten()

    if not cache_wind:
        chrono.start('Dumping windfield to file')
        mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
        chrono.stop()


    # Setting the solver

    t_upper_bound = pb.time_scale if pb.time_scale is not None else 1.2 * pb.geod_l / pb.model.v_a
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

    l = np.linspace(-0.2 * np.pi, 0.1 * np.pi, 10)
    al, au = l[-2], l[-1]
    l2 = np.linspace(al, au, 10)
    for k, a in enumerate(l2):
        s = np.array((np.cos(a), np.sin(a)))
        cost = 'dobrokhodov'
        pn0 = scipy.optimize.brentq(lambda pn:
                                    Utils.power(Utils.airspeed_opti(np.array((pn, 0.)), cost=cost), cost=cost) - pn * (
                                            Utils.airspeed_opti(np.array((pn, 0.)), cost=cost) + s @ pb.model.wind.value(0., pb.x_init)),
                                    1., 300.)
        print(f'pn0 : {pn0:.2f}')
        # pn0 = pn0s[k]
        shooting = Shooting(pb.model.dyn, pb.x_init, 2 * t_upper_bound, mode='energy-opt', domain=pb.domain,
                            energy_ceil=20. * 8.4 * 3.6e6)
        shooting.set_adjoint(-1. * pn0 * s)
        traj = shooting.integrate()
        traj.info = f'trv_opt'
        err = np.min(np.linalg.norm(traj.points - pb.x_target, axis=1) / pb.geod_l)
        reached = err < 0.05
        if reached:
            print(pn0, a)
            trajs.append(traj)

    def auto_upperb(pn):
        va = Utils.airspeed_opti_(pn)
        return 3 * pb.geod_l / va

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
        return np.min(np.linalg.norm(traj.points - pb.x_target) / pb.geod_l)

    candidates =[]

    for pn0 in [5.]:
        print(pn0, end='')
        for factor in np.linspace(0.18, 0.22, 0):
            theta = factor * np.pi
            fb = FixedHeadingFB(pb.model.wind, Utils.airspeed_opti_(pn0), theta, coords=pb.coords)
            psi = fb.value(0, pb.x_init)
            s = np.array((np.cos(psi), np.sin(psi)))
            shooting = Shooting(pb.model.dyn, pb.x_init, auto_upperb(pn0), mode='energy-opt', domain=pb.domain,
                                energy_ceil=20. * 8.4 * 3.6e6)
            shooting.set_adjoint(-1. * pn0 * s)
            traj = shooting.integrate()
            traj.info = f'trv_{int(pn0)}_auto'
            err = np.min(np.linalg.norm(traj.points - pb.x_target, axis=1) / pb.geod_l)
            reached = err < 0.05
            if reached:
                print('|', end='')
                candidates.append((pn0, psi/np.pi))
                trajs.append(traj)
        print()
    print(candidates)
    mdfm.dump_trajs(trajs)

    for v_a in np.linspace(13.19, 14.5, 5):  # [10.15, 11.17, 14.11, 18.51, 22.13]:
        pb.update_airspeed(v_a)
        chrono.start('Computing EF')
        t_upper_bound = 2 * pb.geod_l / pb.model.v_a
        solver_ef = solver = SolverEF(pb, t_upper_bound, max_steps=500, rel_nb_ceil=0.05)
        res = solver.solve()
        chrono.stop()
        print(f'Reach time : {res.duration / 3600:.2f}')
        trajs = solver.get_trajs()
        # mdfm.dump_trajs(trajs)

        chrono.start('Optimal trajectory Extremals')
        pb.load_feedback(FunFB(solver.control))
        sc = TimedSC(pb.model.wind.t_start + res.duration)
        traj = pb.integrate_trajectory(pb.x_init, sc, int_step=res.duration / res.index, t_init=pb.model.wind.t_start)
        traj.type = Utils.TRAJ_OPTIMAL
        traj.info = f'Extremals_{v_a:.2f}'
        chrono.stop()
        mdfm.dump_trajs([traj])


    # Extract information for display and write it to output
    mdfm.ps.load_from_problem(pb)
    mdfm.ps.dump()
    # Also copy the script that produced the result to output dir for later reproduction
    mdfm.save_script(__file__)

    # mdfm.set_case('example_dakar-natal-constr*')
    # mdfm.dump_trajs([res_ef.traj])

    print(f'Results saved to {mdfm.case_dir}')