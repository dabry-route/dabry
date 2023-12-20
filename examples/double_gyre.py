import os

from dabry.misc import Utils, Chrono
from dabry.problem import NPDoubleGyreLi, NPDoubleGyreKularatne, NP3vor
from dabry.solver_ef import SolverEF

if __name__ == '__main__':
    # First set the path to the Dabry root
    os.environ.setdefault('DABRYPATH', '/home/bastien/Documents/work/dabry')
    # Create the double gyre navigation problem
    pb = NPDoubleGyreLi()

    # Initialize the IO module to save output data later
    pb.io.set_case(f'example_double_gyre')
    pb.io.clean_output_dir()

    # Create the solver
    # Give it the navigation problem and a time scale
    solver_ef = SolverEF(pb, max_time=pb.time_scale, dt=pb.time_scale / 100)

    chrono = Chrono()
    chrono.start('Solving problem using extremal field (EF)')
    res_ef = solver_ef.solve()
    chrono.stop()
    if res_ef.status:
        # Solution found
        # Save optimal trajectory
        pb.io.dump_trajs([res_ef.traj])
        print(f'Target reached in : {Utils.time_fmt(res_ef.duration)}')
    else:
        print('No solution found')

    # Save extremal field
    extremals = solver_ef.get_trajs()
    pb.io.dump_trajs(extremals)

    # Save to output directory
    pb.save_ff()  # Flow field
    pb.save_info()  # Problem information
    # Also copy the script that produced the result to output dir for later reproduction
    pb.io.save_script(__file__)

    print(f'Results saved to {pb.io.case_dir}')
