import os
import numpy as np

from dabry.mdf_manager import DDFmanager
from dabry.obstacle import GreatCircleObs, ParallelObs
from dabry.misc import Utils, Chrono
from dabry.problem import IndexedProblem, DatabaseProblem
from dabry.solver_ef import SolverEF
from dabry.solver_rp import SolverRP

if __name__ == '__main__':
    # Choose problem ID for IndexedProblem
    pb_id = 19
    # Or choose database problem. If empty, will use previous ID
    dbpb = ''
    # When running several times, wind data or reachability fronts data can be cached
    cache_wind = False
    cache_rff = False

    # This instance prints absolute elapsed time between operations
    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = DDFmanager(cache_wind=cache_wind, cache_rff=cache_rff)
    mdfm.setup()
    if len(dbpb) > 0:
        case_name = f'example_{dbpb.split("/")[-1]}'
    else:
        case_name = f'example_{IndexedProblem.problems[pb_id][1]}'
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
        pb = DatabaseProblem(dbpb)
    else:
        pb = IndexedProblem(pb_id)

    # pb.flatten()

    if not cache_wind:
        chrono.start('Dumping windfield to file')
        mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
        chrono.stop()

    # Setting the extremal solver
    solver_ef = solver = SolverEF(pb, pb.time_scale)

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
