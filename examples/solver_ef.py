import time
import os

from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem
from mermoz.solver_ef import SolverEF
from mermoz.solver_rp import SolverRP
from mermoz.wind import DiscreteWind


if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 6, 0
    cache = False

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    output_dir = f'/home/bastien/Documents/work/mermoz/output/example_solver-ef_{IndexedProblem.problems[pb_id][1]}'
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
    solver_ef = SolverEF(pb, max_steps=300)

    t_start = time.time()
    solver_ef.solve()
    t_end = time.time()
    time_rp = t_end - t_start

    # solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft, extremals=False)
    # if cache:
    #     solver_rp.rft.load_cache(os.path.join(output_dir, 'rff.h5'))
    # solver_rp.solve()
    # if not cache:
    #     solver_rp.rft.dump_rff(output_dir)

    mdfm.dump_trajs(solver_ef.get_trajs())
    # mdfm.dump_trajs(solver_rp.mp.trajs)

    ps = ParamsSummary()
    ps.set_output_dir(output_dir)
    ps.load_from_problem(pb)
    ps.dump()

    print(f'Results saved to {output_dir}')
