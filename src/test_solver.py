import random

import matplotlib as mpl
import numpy as np

from src.feedback import FixedHeadingFB
from src.mermoz import MermozProblem
from src.model import ZermeloGeneralModel
from src.shooting import Shooting
from src.solver import Solver
from src.stoppingcond import TimedSC
from src.trajectory import Trajectory, AugmentedTraj
from wind import VortexWind, UniformWind

mpl.style.use('seaborn-notebook')


def test_solver():
    # UAV airspeed in m/s
    v_a = 1.
    # UAV goal point x-coordinate in meters
    x_f = 1.
    # The time window upper bound in seconds
    T = 2.5

    offset = np.array([0., 0.])
    omega1 = np.array([0.5, 0.8]) + offset
    omega2 = np.array([0.8, 0.2]) + offset
    omega3 = np.array([0.6, -0.5]) + offset

    vortex1 = VortexWind(omega1[0], omega1[1], -1.)
    vortex2 = VortexWind(omega2[0], omega2[1], -0.5)
    vortex3 = VortexWind(omega3[0], omega3[1], 0.8)
    const_wind = UniformWind(np.array([-0.1, 0.]))

    # Precise the domain for the state. Avoid the center of vortices.
    def domain(x):
        margin = 1e-1  # [m]
        value = True
        value = value and np.linalg.norm(x - vortex1.omega) > margin
        value = value and np.linalg.norm(x - vortex2.omega) > margin
        value = value and np.linalg.norm(x - vortex3.omega) > margin
        value = value and -0.1 < x[0] < 1.5 and -1 < x[1] < 1.
        return value

    total_wind = 3. * const_wind + vortex1 + vortex2 + vortex3

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, x_f)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = np.array([0., 0.])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, domain=domain, visual_mode='full')
    mp.display.set_wind_density(2)

    solver = Solver(mp, x_init, T, n_min_opti=3)
    solver.solve()


if __name__ == '__main__':
    test_solver()