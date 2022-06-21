import time

from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem, problems
from mermoz.solver_rp import SolverRP

if __name__ == '__main__':
    # Choose problem ID
    pb_id = 3
    seed = 20

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    output_dir = f'/home/bastien/Documents/work/mermoz/output/example_solver-rp_{problems[pb_id][1]}_{seed}'
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir()

    # Create problem
    pb = IndexedProblem(pb_id, seed=seed)

    # Log windfield to file
    mdfm.dump_wind(pb.model.wind, nx=51, ny=51, bl=pb.bl, tr=pb.tr)

    # Setting the solver
    nx_rft = 51
    ny_rft = 51
    nt_rft = 25
    solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft)

    t_start = time.time()
    solver_rp.solve()
    t_end = time.time()
    time_rp = t_end - t_start

    solver_rp.rft.dump_rff(output_dir)
    mdfm.dump_trajs(solver_rp.mp.trajs)
    mdfm.dump_trajs(solver_rp.mp_pmp.trajs)

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_solver_rp(solver_rp)
    ps.dump()
