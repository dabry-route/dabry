import os
import numpy as np

from dabry.ddf_manager import DDFmanager
from dabry.model import ZermeloGeneralModel
from dabry.obstacle import GreatCircleObs, ParallelObs
from dabry.misc import Utils, Chrono
from dabry.problem import IndexedProblem, DatabaseProblem, NavigationProblem
from dabry.solver_ef import SolverEF
from dabry.solver_rp import SolverRP
from dabry.feedback import GSTargetFB
from dabry.stoppingcond import DistanceSC
from dabry.wind import DiscreteWind

if __name__ == '__main__':

    cache_wind = False
    cache_rff = False

    # This instance prints absolute elapsed time between operations
    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = DDFmanager(cache_wind=cache_wind, cache_rff=cache_rff)
    mdfm.setup()
    mdfm.set_case('test_ecmwf')
    mdfm.clean_output_dir()

    # Space and time discretization
    # Will be used to save wind when wind is analytical and shall be sampled
    # Will also be used by front tracking module
    nx_rft = 101
    ny_rft = 101
    nt_rft = 20

    wind = DiscreteWind()
    wind.load_from_cds('/home/bastien/Documents/work/zermelo_earth/test.grb2',
                       np.array([140, -60]),
                       np.array([175, -30])
                       )
    zermelo_model = ZermeloGeneralModel(23, coords=Utils.COORD_GCS)
    zermelo_model.update_wind(wind)
    pb = NavigationProblem(zermelo_model, Utils.DEG_TO_RAD * np.array([150, -50]),
                           Utils.DEG_TO_RAD * np.array([170, -40]), coords=Utils.COORD_GCS)

    # pb.flatten()

    if not cache_wind:
        chrono.start('Dumping windfield to file')
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
    chrono.stop()

    # Setting the extremal solver
    solver_ef = solver = SolverEF(pb, pb.time_scale, max_steps=700, rel_nb_ceil=0.02, quick_solve=True)

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

    # pb.flatten()
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
        #mdfm.dump_trajs([res_rp.traj])
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
