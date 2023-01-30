import os
from dabry.mdf_manager import DDFmanager
from dabry.misc import *
from dabry.problem import IndexedProblem, DatabaseProblem
from dabry.shooting import Shooting
from dabry.solver_ef import SolverEF, Pareto


if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 5, 0
    dbpb = '44W_16S_9W_25S_20220301_12'  # '72W_15S_0W_57S_20220301_12' # '37W_8S_16W_17S_20220301_12'
    cache_rff = False
    cache_wind = False

    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = DDFmanager(cache_wind=True)
    mdfm.setup()
    if dbpb is not None:
        case_name = f'example_energy_{dbpb}'
    else:
        case_name = f'example_energy_{IndexedProblem.problems[pb_id][1]}'
    mdfm.set_case(case_name)
    mdfm.clean_output_dir()

    nx_rft, ny_rft, nt_rft = 101, 101, 20

    if len(dbpb) > 0:
        pb = DatabaseProblem(os.path.join('/home/bastien/Documents/data/wind/ncdc/', dbpb, 'wind.h5'),
                             x_init=DEG_TO_RAD * np.array([-35.2080905, -5.805398]),
                             x_target=DEG_TO_RAD * np.array([-17.447938, 14.693425]))
    else:
        pb = IndexedProblem(pb_id, seed=seed)

    chrono.start('Dumping windfield to file')
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
    chrono.stop()

    pareto = Pareto()
    pareto.load(mdfm.case_dir)

    asp_inits = [[24],
                 [21]]

    t_upper_bound = 1.5 * pb.geod_l / min(asp_inits[0] + asp_inits[1])
    dt = t_upper_bound / 1000
    for mode in [0, 1]:
        for asp_init in asp_inits[mode]:
            chrono.start(f'Computing EF Mode {mode} Airspeed {asp_init:.2f}')
            pb.update_airspeed(asp_init)
            solver_ef = solver = SolverEF(pb, t_upper_bound, mode=mode, rel_nb_ceil=0.02,
                                          dt=dt,
                                          no_coll_filtering=True,
                                          asp_init=asp_init,
                                          quick_solve=not mode,
                                          pareto=None if not mode else pareto)
            optim_res = solver.solve()
            reach_time = optim_res.duration
            reach_cost = optim_res.cost
            reach_time_tef = reach_time
            chrono.stop()
            print(f'Time/Cost : {reach_time / 3600:.2f}h/{reach_cost / 3.6e6:.2f}kWh')
            if mode == 0:
                mdfm.dump_trajs([optim_res.traj])
                if optim_res.status:
                    pareto.add((optim_res.duration, optim_res.cost))
            else:
                for k in optim_res.bests.keys():
                    mdfm.dump_trajs([optim_res.trajs[k]])
                    if optim_res.status:
                        pareto.add((optim_res.bests[k].t - pb.model.wind.t_start, optim_res.bests[k].cost))
            # trajs = solver.get_trajs()
            # mdfm.dump_trajs(trajs)

    pareto.dump(mdfm.case_dir)

    # chrono.start('Computing Energy EF')
    # t_upper_bound = 3 * pb.geod_l / pb.aero.v_minp
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
    # err = np.min(np.linalg.norm(traj.points - pb.x_target, axis=1) / pb.geod_l)
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
        t_upper_bound = 2 * pb.geod_l / pb.model.v_a
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
    sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), 0.02 * pb.geod_l)
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
    #     t_upper_bound = pb.time_scale if pb.time_scale is not None else 1.2 * pb.geod_l / pb.model.v_a
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
    #     sc = DistanceSC(lambda x: pb.distance(x, pb.x_init), pb.geod_l * 0.01)
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
    # sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), pb.geod_l / 100.)
    # traj = pb.integrate_trajectory(pb.x_init, sc, max_iter=2 * iit, int_step=reach_time / iit,
    #                                t_init=pb.model.wind.t_start)
    # traj.info = 'GSTarget FB'
    # chrono.stop()
    #
    # chrono.start('HTarget FB')
    # pb.load_feedback(HTargetFB(pb.x_target, pb.coords))
    # sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), pb.geod_l / 100.)
    # traj = pb.integrate_trajectory(pb.x_init, sc, max_iter=2 * iit, int_step=reach_time / iit,
    #                                t_init=pb.model.wind.t_start)
    # traj.info = 'HTarget FB'
    # chrono.stop()

    # pb.load_feedback(GreatCircleFB(pb.model.wind, pb.model.v_a, pb.x_target))
    # sc = DistanceSC(lambda x: pb.distance(x, pb.x_target), pb.geod_l / 100.)
    # traj = pb.integrate_trajectory(pb.x_init, sc, max_iter=2 * iit, int_step=reach_time / iit, t_init=t_init)
    # traj.info = 'Great circle'

    # chrono.start('Dumping data')
    #
    # mdfm.dump_trajs(pb.trajs)
    # chrono.stop()

    # Extract information for display and write it to output
    mdfm.ps.load_from_problem(pb)
    mdfm.ps.dump()
    # Also copy the script that produced the result to output dir for later reproduction
    mdfm.save_script(__file__)

    print(f'Results saved to {mdfm.case_dir}')
