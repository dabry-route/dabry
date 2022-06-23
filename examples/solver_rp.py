import time
import os

from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem, problems
from mermoz.solver_rp import SolverRP
from mermoz.wind import DiscreteWind

if __name__ == '__main__':
    # Choose problem ID
    pb_id = 1
    seed = 0
    cache = False

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    output_dir = f'/home/bastien/Documents/work/mermoz/output/example_solver-rp_{problems[pb_id][1]}'
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir(keep_rff=cache)

    nx_rft = 101
    ny_rft = 101
    nt_rft = 20

    # Create problem
    pb = IndexedProblem(pb_id, seed=seed)
    # d_wind = DiscreteWind(interp='pwc')
    # d_wind.load_from_wind(pb.model.wind, nx_rft, ny_rft, pb.bl, pb.tr, pb.coords)
    # pb.model.wind = d_wind

    # Log windfield to file
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, bl=pb.bl, tr=pb.tr)

    # Setting the solver
    solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft, nt_pmp=1000)
    if cache:
        solver_rp.rft.load_cache(os.path.join(output_dir, 'rff.h5'))

    t_start = time.time()
    solver_rp.solve()
    t_end = time.time()
    time_rp = t_end - t_start

    solver_rp.mp.trajs.append(solver_rp.rft.backward_traj(pb.x_target, pb.x_init, solver_rp.opti_ceil,
                                                          solver_rp.T, pb.model, N_disc=1000))

    # solver_rp.rft.build_front(pb.x_target)

    if not cache:
        solver_rp.rft.dump_rff(output_dir)
    mdfm.dump_trajs(solver_rp.mp.trajs)
    mdfm.dump_trajs(solver_rp.mp_pmp.trajs)

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_solver_rp(solver_rp)
    ps.dump()
