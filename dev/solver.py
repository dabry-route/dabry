import os
import numpy as np

from dabry.mdf_manager import DDFmanager
from dabry.obstacle import GreatCircleObs, ParallelObs, MaxiObs, LSEMaxiObs
from dabry.misc import Utils, Chrono
from dabry.problem import IndexedProblem, DatabaseProblem
from dabry.solver_ef import SolverEF
from dabry.solver_rp import SolverRP

if __name__ == '__main__':
    # Choose problem ID for IndexedProblem
    pb_id = 21
    # Or choose database problem. If empty, will use previous ID
    dbpb = '' # 'ncdc/44W_16S_9W_25S_20220301_12'
    suffix = ''
    # When running several times, wind data or reachability fronts data can be cached
    cache_wind = True
    cache_rff = False

    # This instance prints absolute elapsed time between operations
    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = DDFmanager(cache_wind, cache_rff)
    mdfm.setup()
    if len(dbpb) > 0:
        case_name = f'example_solver-ef_{dbpb}' + (f'_{suffix}' if len(suffix) > 0 else '')
    else:
        case_name = f'{pb_id}_{IndexedProblem.problems[pb_id][1]}' + (f'_{suffix}' if len(suffix) > 0 else '')
    mdfm.set_case(case_name)
    mdfm.clean_output_dir()

    # Space and time discretization
    # Will be used to save wind when wind is analytical and shall be sampled
    # Will also be used by front tracking module
    nx_rft = 101
    ny_rft = 101
    nt_rft = 20

    # Create problem
    if len(dbpb) > 0:
        obs = []
        # obs.append(GreatCircleObs(np.array((-17 * DEG_TO_RAD, 0 * DEG_TO_RAD)),
        #                           np.array((-17 * DEG_TO_RAD, 10 * DEG_TO_RAD))))
        # obs.append(GreatCircleObs(np.array((-17 * DEG_TO_RAD, 0 * DEG_TO_RAD)),
        #                           np.array((-17 * DEG_TO_RAD, 10 * DEG_TO_RAD))))
        # obs.append(ParallelObs(18 * DEG_TO_RAD, True))
        # obs1 = GreatCircleObs(np.array((-30 * DEG_TO_RAD, 0 * DEG_TO_RAD)),
        #                           np.array((-30 * DEG_TO_RAD, -1 * DEG_TO_RAD)))
        # obs2 = GreatCircleObs(np.array((-30 * DEG_TO_RAD, 10 * DEG_TO_RAD)),
        #                           np.array((-31 * DEG_TO_RAD, 10 * DEG_TO_RAD)))
        # obs.append(LSEMaxiObs([obs1, obs2]))
        pb = DatabaseProblem(dbpb,
                             airspeed=23.,
                             obstacles=obs,
                             x_init=Utils.DEG_TO_RAD * np.array([-35.2080905, -5.805398]),
                             x_target=Utils.DEG_TO_RAD * np.array([-17.447938, 14.693425]))
    else:
        pb = IndexedProblem(pb_id)

    # pb.flatten()

    chrono.start('Dumping windfield to file')
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
    chrono.stop()

    mdfm.dump_obs(pb)
    pb.update_airspeed(20.)

    # Setting the extremal solver
    t_upper_bound = pb.time_scale if pb.time_scale is not None else pb.l_ref / pb.model.v_a
    solver_ef = solver = SolverEF(pb, t_upper_bound, max_steps=700, rel_nb_ceil=0.02,
                                  quick_solve=True, mode=0)

    chrono.start('Solving problem using extremal field (EF)')
    res_ef = solver_ef.solve()
    chrono.stop()
    if res_ef.status:
        # Solution found
        # Save optimal trajectory
        mdfm.dump_trajs([res_ef.traj])

    else:
        print('No solution found')

    # Save extremal field for display purposes
    extremals = solver_ef.get_trajs()
    mdfm.dump_trajs(extremals)

    """
    # Setting the front tracking solver
    solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft)
    if cache_rff:
        solver_rp.rft.load_cache(os.path.join(mdfm.case_dir, 'rff.h5'))

    chrono.start('Solving problem using reachability front tracking (RFT)')
    res_rp = solver_rp.solve()
    chrono.stop()
    if res_rp.status:
        # Save optimal trajectory
        mdfm.dump_trajs([res_rp.traj])
        print(f'Target reached in : {time_fmt(res_rp.duration)}')

    # Save fronts for display purposes
    if not cache_rff:
        solver_rp.rft.dump_rff(mdfm.case_dir)
    """

    # Extract information for display and write it to output
    mdfm.ps.load_from_problem(pb)
    mdfm.ps.dump()
    # Also copy the script that produced the result to output dir for later reproduction
    mdfm.save_script(__file__)

    print(f'Results saved to {mdfm.case_dir}')
