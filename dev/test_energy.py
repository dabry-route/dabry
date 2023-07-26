import numpy as np

from dabry.aero import MermozAero
from dabry.ddf_manager import DDFmanager
from dabry.misc import Utils, Chrono
from dabry.problem import IndexedProblem, DatabaseProblem
from dabry.shooting import Shooting
from dabry.solver_ef import SolverEF, Pareto


if __name__ == '__main__':
    cache_wind = False

    case = 'movor'

    ddf = DDFmanager(cache_wind=cache_wind)
    ddf.setup()
    ddf.set_case(case)
    ddf.clean_output_dir()

    pb = IndexedProblem('movor')
    if not cache_wind:
        ddf.dump_wind(pb.model.wind, nx=101, ny=101, nt=20, bl=pb.bl, tr=pb.tr)

    pb.aero = MermozAero()
    ddf.log(pb)

    chrono = Chrono()

    # pareto = Pareto()
    # pareto.load(ddf.case_dir)

    kwargs = {
        'max_steps': 500,
        'mode': 1,
        'max_active_ext': 5000,
        'v_bounds': (18, 20),
        'N_pn_init': 4,
        'rel_nb_ceil': 0.1,
        'rel_cost_ceil': 0.2,
        'quick_solve': True,
        'quick_offset': 20
    }
    solver = SolverEF(pb, **kwargs)
    res_ef = solver.solve(verbose=2)
    ddf.dump_trajs(solver.get_trajs())
    ddf.dump_trajs(res_ef.trajs)
    for v_a in [18, 19, 20, 21, 22]:
        kwargs = {
            'max_steps': 500,
            'mode': 0,
            'max_active_ext': 5000,
            'rel_nb_ceil': 0.1,
            'rel_cost_ceil': 0.2,
            'quick_solve': True
        }
        pb.update_airspeed(v_a)
        solver = SolverEF(pb, **kwargs)
        res_ef = solver.solve(verbose=2)
        # ddf.dump_trajs(solver.get_trajs())
        ddf.dump_trajs([res_ef.traj])
