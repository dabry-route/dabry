import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.solver import Solver
from mermoz.wind import LinearWind

mpl.style.use('seaborn-notebook')


def example_linear_wind():
    """
    Example of the PMP-based solver for the time-optimal Zermelo
    control problem with a linear windfield
    """
    # UAV airspeed in m/s
    v_a = 0.5
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
                    np.pi / 16. - 5e-2,
                    N_disc_init=2,
                    opti_ceil=5e-3,
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
    example_linear_wind()
    plt.show()
