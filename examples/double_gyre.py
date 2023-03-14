from dabry.ddf_manager import DDFmanager
from dabry.misc import Utils, Chrono
from dabry.problem import IndexedProblem
from dabry.solver_ef import SolverEF
from dabry.solver_rp import SolverRP

if __name__ == '__main__':
    # Create a navigation problem from list of reference problems
    pb_id = 4
    pb = IndexedProblem(pb_id)

    # Create the logger which will save solver information
    mdfm = DDFmanager()
    mdfm.setup()
    case_name = f'example_{IndexedProblem.problems[pb_id][1]}'
    mdfm.set_case(case_name)
    mdfm.clean_output_dir()
    # Save the windfield to file
    nx, ny, nt = 101, 101, 20
    mdfm.dump_wind(pb.model.wind, nx=nx, ny=ny, nt=nt, bl=pb.bl, tr=pb.tr)

    # Create the solver
    # Give it the navigation problem and a time scale
    solver_ef = solver = SolverEF(pb, pb.time_scale)

    chrono = Chrono()
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

    # Save extremal field
    extremals = solver_ef.get_trajs()
    mdfm.dump_trajs(extremals)

    # Change this to allow front tracking
    front_tracking = False
    if front_tracking:
        # Setting the front tracking solver
        solver_rp = SolverRP(pb, nx, ny, nt)

        chrono.start('Solving problem using reachability front tracking (RFT)')
        res_rp = solver_rp.solve()
        chrono.stop()
        if res_rp.status:
            # Save optimal trajectory
            mdfm.dump_trajs([res_rp.traj])
            print(f'Target reached in : {Utils.time_fmt(res_rp.duration)}')

        # Save fronts
        solver_rp.rft.dump_rff(mdfm.case_dir)

    # Extract information for display and write it to output
    mdfm.ps.load_from_problem(pb)
    mdfm.ps.dump()
    # Also copy the script that produced the result to output dir for later reproduction
    mdfm.save_script(__file__)

    print(f'Results saved to {mdfm.case_dir}')
