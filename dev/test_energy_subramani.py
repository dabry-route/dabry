import numpy as np

from dabry.aero import MermozAero, SubramaniAero
from dabry.ddf_manager import DDFmanager
from dabry.misc import Utils, Chrono
from dabry.model import ZermeloGeneralModel
from dabry.problem import IndexedProblem, DatabaseProblem, NavigationProblem
from dabry.shooting import Shooting
from dabry.solver_ef import SolverEF, Pareto
from dabry.wind import DoubleGyreDampedWind, DiscreteWind

if __name__ == '__main__':
    cache_wind = False

    f = 1e6
    wind = DoubleGyreDampedWind(0 * f, 0 * f, 2 * f, 1 * f, 2, 0.5 * f, 0.5 * f, 0.8 * f, 1 * f)
    x_init = f * np.array((0.2, 0.2))
    x_target = f * np.array((0.6, 0.6))
    bl = f * np.array((0, 0))
    tr = f * np.array((1, 1))
    model = ZermeloGeneralModel(0.5, 'cartesian')
    disc_wind = DiscreteWind()
    disc_wind.load_from_wind(wind, 101, 101, bl, tr, 'cartesian', fd=True)
    model.update_wind(disc_wind)
    pb = NavigationProblem(model, x_init, x_target, 'cartesian', bl=bl, tr=tr)
    ddf = DDFmanager()
    ddf.setup()
    ddf.set_case('test_energy')
    ddf.clean_output_dir()
    ddf.dump_wind(pb.model.wind, nx=101, ny=101, nt=20, bl=pb.bl, tr=pb.tr)

    pb.aero = SubramaniAero()
    ddf.log(pb)

    chrono = Chrono()

    # pareto = Pareto()
    # pareto.load(ddf.case_dir)

    kwargs = {
        'max_steps': 500,
        'mode': 1,
        'max_active_ext': 5000,
        'v_bounds': (0.2, 0.6),
        'N_pn_init': 4,
        'rel_nb_ceil': 0.1,
        'rel_cost_ceil': 0.2,
        'quick_solve': True,
        'quick_offset': 20,
        'alpha_factor': 0.75,
    }

    solver = SolverEF(pb, **kwargs)
    res_ef = solver.solve(verbose=2)
    ddf.dump_trajs(solver.get_trajs())
    ddf.dump_trajs(res_ef.trajs.values())

    for v_a in [0.2, 0.3, 0.4, 0.5, 0.6]:
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
