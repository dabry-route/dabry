import numpy as np

from dabry.ddf_manager import DDFmanager
from dabry.model import ZermeloGeneralModel
from dabry.obstacle import CircleObs
from dabry.misc import Utils, Chrono
from dabry.problem import NavigationProblem
from dabry.solver_ef import SolverEF

if __name__ == '__main__':

    # This instance prints absolute elapsed time between operations
    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = DDFmanager()
    mdfm.setup()
    case_name = 'obstacle_without_scheme'
    mdfm.set_case(case_name)
    mdfm.clean_output_dir()

    # Space and time discretization
    # Will be used to save wind when wind is analytical and shall be sampled
    # Will also be used by front tracking module
    nx_rft = 300
    ny_rft = 300
    nt_rft = 20
    model = ZermeloGeneralModel(23.)
    obstacles = [CircleObs(np.array((0.5, 0.)), 0.2)]
    pb = NavigationProblem(model, np.zeros(2), np.array((1., 0.)), Utils.COORD_CARTESIAN,
                           bl=np.array((-0.2, -0.5)), tr=np.array((1.2, 0.5)), obstacles=obstacles)
    # pb.flatten()

    chrono.start('Dumping windfield to file')
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
    chrono.stop()

    mdfm.dump_obs(pb)

    # Setting the extremal solver
    t_upper_bound = pb.l_ref / pb.model.v_a
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
