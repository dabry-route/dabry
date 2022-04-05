import os
import random

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

from mermoz.mdf_manager import MDFmanager
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.solver import Solver
from mermoz.wind import VortexWind, UniformWind, LinearWind, DiscreteWind, RankineVortexWind
from mermoz.misc import *

mpl.style.use('seaborn-notebook')


def test_solver():
    output_dir = '/home/bastien/Documents/work/mermoz/output/example_solver'
    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)

    # UAV airspeed in m/s
    v_a = 23.

    # The time window upper bound in seconds
    T = 20 * 3600

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
    print(vortex1.max_speed())
    print(vortex2.max_speed())
    print(vortex3.max_speed())
    const_wind = UniformWind(np.array([0., 0.]))  # UniformWind(np.array([factor_speed * -0.1, 0.]))

    # Wind array boundaries
    bl = 1e6 * np.array([-0.2, -1.])
    tr = 1e6 * np.array([1.2, 1.])

    # Precise the domain for the state. Avoid the center of vortices.
    def domain(x):
        margin = factor * 1e-1  # [m]
        value = True
        value = value and np.linalg.norm(x - vortex1.omega) > margin
        value = value and np.linalg.norm(x - vortex2.omega) > margin
        value = value and np.linalg.norm(x - vortex3.omega) > margin
        value = value and bl[0] < x[0] < tr[0] and bl[1] < x[1] < tr[1]
        return value

    alty_wind = 3. * const_wind + vortex1 + vortex2 + vortex3
    total_wind = DiscreteWind()
    total_wind.load_from_wind(alty_wind, 101, 101, bl, tr, coords='cartesian')
    mdfm.dump_wind(total_wind)

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = factor * np.array([0., 0.])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, domain=domain)

    # u_min = -3 * np.pi / 8.
    # u_max = 3 * np.pi / 8.
    # u_min = - np.pi / 8.
    # u_max = 0.
    u_min = -np.pi / 2 + np.pi / 6 * random.random()
    u_max = np.pi / 2 - np.pi / 6 * random.random()

    solver = Solver(mp,
                    x_init,
                    x_target,
                    T,
                    u_min,
                    u_max,
                    output_dir,
                    N_disc_init=2,
                    opti_ceil=factor / 20,
                    neighb_ceil=factor / 40,
                    n_min_opti=1,
                    adaptive_int_step=False,
                    N_iter=10000)
    solver.log_config()

    solver.setup()
    solver.solve_fancy()

    mdfm.dump_trajs(mp.trajs)


if __name__ == '__main__':
    test_solver()
