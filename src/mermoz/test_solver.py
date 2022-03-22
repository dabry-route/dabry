import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

from .problem import MermozProblem
from .model import ZermeloGeneralModel
from .solver import Solver
from .wind import VortexWind, UniformWind, LinearWind

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
    mp = MermozProblem(zermelo_model, T=T, domain=domain, visual_mode='only-map')
    mp.display.set_wind_density(2)

    solver = Solver(mp,
                    x_init,
                    np.array([x_f, 0.]),
                    T,
                    -3 * np.pi / 8.,
                    3 * np.pi / 8.,
                    N_disc_init=10,
                    opti_ceil=5e-2,
                    neighb_ceil=1e-1,
                    n_min_opti=2,
                    adaptive_int_step=False)
    solver.log_config()

    solver.setup()
    solver.solve()


def test_solver2():
    # UAV airspeed in m/s
    v_a = 1.
    # UAV goal point x-coordinate in meters
    x_f = 1.
    # The time window upper bound in seconds
    T = 2.
    # The wind gradient
    gradient = np.array([[0., v_a / 10.],
                         [0., 0.]])
    origin = np.array([0., 0.])
    value_origin = np.array([0., 0.])

    total_wind = LinearWind(gradient, origin, value_origin)

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, x_f)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = np.array([0., 0.])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map', axes_equal=False)
    mp.display.set_wind_density(2)

    solver = Solver(mp,
                    x_init,
                    np.array([x_f, 0.]),
                    T,
                    0.,
                    np.pi / 16.,
                    N_disc_init=2,
                    opti_ceil=1e-3,
                    neighb_ceil=1e-4,
                    n_min_opti=1,
                    adaptive_int_step=False)
    solver.setup()
    solver.solve()

    # Analytic optimal trajectory
    w = -gradient[0, 1]
    theta_f = 0.01

    def analytic_traj(theta, theta_f):
        x = 0.5 * v_a / w * (-1 / np.cos(theta_f) * (np.tan(theta_f) - np.tan(theta)) +
                             np.tan(theta) * (1 / np.cos(theta_f) - 1 / np.cos(theta)) -
                             np.log((np.tan(theta_f)
                                     + 1 / np.cos(theta_f)) / (
                                            np.tan(theta) + 1 / np.cos(theta))))
        y = v_a / w * (1 / np.cos(theta) - 1 / np.cos(theta_f))
        return x + x_f, y

    def residual(theta_f):
        return analytic_traj(-theta_f, theta_f)[0]

    theta_f = scipy.optimize.newton_krylov(residual, 0.01)
    print(f'theta_f : {theta_f}')

    points = np.array(list(map(lambda theta: analytic_traj(theta, theta_f), np.linspace(-theta_f, theta_f, 1000))))
    solver.mp.display.map.plot(points[:, 0], points[:, 1], label='analytic')
    solver.mp.display.map.legend()


if __name__ == '__main__':
    test_solver()
    plt.show()
