import os
from datetime import datetime

import numpy as np

from dabry.ddf_manager import DDFmanager
from dabry.obstacle import GreatCircleObs, ParallelObs
from dabry.misc import Utils, Chrono
from dabry.problem import IndexedProblem, DatabaseProblem
from dabry.solver_ef import SolverEF
from dabry.solver_rp import SolverRP
from dabry.feedback import GSTargetFB
from dabry.stoppingcond import DistanceSC

if __name__ == '__main__':
    # Choose problem ID for IndexedProblem
    pb_id = 5
    # Or choose database problem. If empty, will use previous ID
    dbpb = True
    # When running several times, wind data or reachability fronts data can be cached
    cache_wind = False
    cache_rff = False

    # This instance prints absolute elapsed time between operations
    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = DDFmanager(cache_wind=cache_wind, cache_rff=cache_rff)
    mdfm.setup()
    mdfm.set_case('solver_example')
    mdfm.clean_output_dir()

    # Space and time discretization
    # Will be used to save wind when wind is analytical and shall be sampled
    # Will also be used by front tracking module
    nx_rft = 101
    ny_rft = 101
    nt_rft = 20

    # Create problem
    if dbpb:
        pb = DatabaseProblem(x_init=Utils.DEG_TO_RAD * np.array([-17.447938, 14.693425]),
                             x_target=Utils.DEG_TO_RAD * np.array([-35.2080905, -5.805398]),
                             airspeed=23,
                             t_start=datetime(2021, 11, 1, 12, 0).timestamp(),
                             t_end=datetime(2021, 11, 3, 12, 0).timestamp())
    else:
        pb = IndexedProblem(pb_id)

    # pb.flatten()

    if not cache_wind:
        chrono.start('Dumping windfield to file')
        mdfm.dump_wind(pb.model.ff, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
        chrono.stop()

    # Setting the extremal solver
    solver_ef = solver = SolverEF(pb, pb.time_scale, max_steps=700, rel_nb_ceil=0.01, quick_solve=True)


    chrono.start('Solving problem using extremal field (EF)')
    res_ef = solver_ef.solve()
    chrono.stop()
    if res_ef.status:
        # Solution found
        # Save optimal trajectory
        mdfm.dump_trajs([res_ef.traj])
        print(f'Target reached in : {Utils.time_fmt(res_ef.duration)}')
    else:
        print('No solution found')
    

    # Save extremal field for display purposes
    extremals = solver_ef.get_trajs()
    mdfm.dump_trajs(extremals)
    mdfm.dump_obs(pb, nx_rft, ny_rft)

    pb.orthodromic()
    mdfm.dump_trajs([pb.trajs[-1]])


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
        print(f'Target reached in : {Utils.time_fmt(res_rp.duration)}')

    # Save fronts for display purposes
    if not cache_rff:
        solver_rp.rft.dump_rff(mdfm.case_dir)
    """

    # Extract information for display and write it to output
    mdfm.ps.load_from_problem(pb)
    mdfm.ps.dump()
    # Also copy the script that produced the result to output dir for later reproduction
    mdfm.save_script(__file__)

    # mdfm.set_case('example_dakar-natal-constr*')
    # mdfm.dump_trajs([res_ef.traj])

    print(f'Results saved to {mdfm.case_dir}')
