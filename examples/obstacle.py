import time
import os

from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem
from mermoz.solver import Solver
from mermoz.wind import DiscreteWind
from mermoz.feedback import ConstantFB
from mermoz.stoppingcond import TimedSC

if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 10, 0

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    output_dir = f'/home/bastien/Documents/work/mermoz/output/example_solver-rp_{IndexedProblem.problems[pb_id][1]}'
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir()

    # Create problem
    pb = IndexedProblem(pb_id)

    # Log windfield to file
    nx_wind, ny_wind = 101, 101
    mdfm.dump_wind(pb.model.wind, nx=101, ny=101, bl=pb.bl, tr=pb.tr)

    u_min, u_max = np.pi/2 * 0.15, np.pi/2.*0.2
    # Setting the solver
    solver = Solver(pb, pb.x_init, pb.x_target, pb._geod_l / pb.model.v_a, u_min, u_max, output_dir, N_disc_init=100, max_shoot=0)

    t_start = time.time()
    solver.solve_fancy()
    t_end = time.time()
    time_rp = t_end - t_start


    pb.load_feedback(ConstantFB(0.01))
    pb.integrate_trajectory(pb.x_init, TimedSC(pb._geod_l / pb.model.v_a), max_iter=100, int_step=pb._geod_l / pb.model.v_a / 100)

    mdfm.dump_trajs(pb.trajs)

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    ps.dump()

    print(f'Results saved to {output_dir}')
