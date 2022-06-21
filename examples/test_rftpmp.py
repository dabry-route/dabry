import os
import random
import time

import matplotlib as mpl
import numpy as np
from math import atan, cos, sin, atan2
from pyproj import Proj

from mermoz.feedback import FixedHeadingFB, TargetFB, ConstantFB
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.solver import Solver
from mermoz.solver_rp import SolverRP
from mermoz.stoppingcond import TimedSC, DistanceSC
from mermoz.wind import VortexWind, UniformWind, LinearWind, DiscreteWind, RankineVortexWind
from mermoz.misc import *
from mermoz.rft import RFT
from mdisplay.geodata import GeoData
from math import asin

mpl.style.use('seaborn-notebook')


def run():
    output_dir = '/home/bastien/Documents/work/mermoz/output/example_test_rftpmp'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)

    coords = COORD_CARTESIAN

    # UAV airspeed in m/s
    v_a = 23.

    x_f = 1e6
    x_target = np.array([x_f, 0.])

    # Wind array boundaries
    bl = 1e6 * np.array([-0.2, -1.])
    tr = 1e6 * np.array([1.2, 1.])

    factor = 1e6
    factor_speed = 23.
    x_f = factor * 1.
    x_target = np.array([x_f, 0.])

    offset = np.array([0., 0.])
    omega1 = factor * (np.array([0.5, 0.8]) + offset)
    omega2 = factor * (np.array([0.8, 0.2]) + offset)
    omega3 = factor * (np.array([0.6, -0.5]) + offset)

    vortex1 = RankineVortexWind(omega1[0], omega1[1], factor * factor_speed * -1., factor * 1e-1)
    vortex2 = RankineVortexWind(omega2[0], omega2[1], factor * factor_speed * -0.8, factor * 1e-1)
    vortex3 = RankineVortexWind(omega3[0], omega3[1], factor * factor_speed * 0.8, factor * 1e-1)
    const_wind = UniformWind(np.array([0., 0.]))

    alty_wind = 3. * const_wind + vortex1 + vortex2 + vortex3
    total_wind = DiscreteWind()
    total_wind.load_from_wind(alty_wind, 101, 101, bl, tr, coords='cartesian')
    pmp_wind = DiscreteWind()
    pmp_wind.load_from_wind(-1. * alty_wind, 101, 101, bl, tr, coords='cartesian')
    mdfm.dump_wind(total_wind)

    x_init = np.array((0., 0.))

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
    zermelo_model.update_wind(total_wind)

    mp = MermozProblem(zermelo_model, coords=coords, autodomain=True, mask_land=False)
    nx_rft = 51
    ny_rft = 51
    nt_rft = 15
    solver_rp = SolverRP(mp, x_init, x_target, nx_rft, ny_rft, nt_rft)

    solver_rp.solve()

    #mp.load_feedback(TargetFB(mp.model.wind, v_a, x_target, mp.coords))
    #mp.integrate_trajectory(x_init, DistanceSC(lambda x: distance(x, x_target, coords=coords), opti_ceil), int_step=T / (nt_pmp - 1))

    mdfm.dump_trajs(mp.trajs)
    mdfm.dump_trajs(solver_rp.mp_pmp.trajs)
    solver_rp.rft.dump_rff(output_dir)

    ps = ParamsSummary({}, output_dir)
    ps.load_from_solver(solver_rp)
    # ps.add_param('pmp_time', time_pmp)
    ps.dump()


if __name__ == '__main__':
    run()
