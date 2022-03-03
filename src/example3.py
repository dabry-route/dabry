import time

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from src.feedback import FixedHeadingFB
from src.mermoz import MermozProblem
from src.model import ZermeloGeneralModel
from src.shooting import Shooting
from src.stoppingcond import TimedSC
from wind import VortexWind, UniformWind, RealWind

mpl.style.use('seaborn-notebook')


def example3():
    """
    Example of the shooting method on a vortex wind
    """
    print("Example 2")
    print("Building model... ", end='')
    t_start = time.time()
    # UAV airspeed in m/s
    v_a = 23.
    # UAV goal point x-coordinate in meters
    x_f = 1.
    # The time window upper bound in seconds
    T = 1 / 23.


    # Precise the domain for the state. Avoid the center of vortices.
    def domain(x):
        # Box
        eps = 1e-3
        b4 = (0 - eps < x[0] < 1 + eps and -1 - eps < x[1] < 1. + eps)
        return b4

    total_wind = RealWind('../../extractWindData/saved_wind/Vancouver-Honolulu-1.0')

    print(total_wind.value(np.array([0.5, 0.5])))

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, x_f)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = np.array([0.99, 0.99])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map')
    mp.display.set_wind_density(3)
    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")

    # Set a list of initial adjoint states for the shooting method
    initial_headings = np.linspace(np.pi, 3*np.pi/2, 20)
    list_p = list(map(lambda theta: -np.array([np.cos(theta), np.sin(theta)]), initial_headings))

    print(f"Shooting PMP trajectories ({len(list_p)})... ", end='')
    t_start = time.time()

    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined

    for k, p in enumerate(list_p):
        shoot = Shooting(zermelo_model.dyn, x_init, T, adapt_ts=False, N_iter=100, domain=domain)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)

    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")

    # Get also explicit control law traejctories
    # Here we add trajectories trying to steer the UAV
    # on a straight line starting from (0, 0) and with a given
    # heading angle
    list_headings = np.linspace(np.pi - 0.2, np.pi + 0.2, 1)
    print(f"Shooting explicit control laws ({list_headings.size})... ", end='')
    t_start = time.time()
    for heading in list_headings:
        mp.load_feedback(FixedHeadingFB(mp._model.wind, v_a, heading))
        mp.integrate_trajectory(x_init, TimedSC(T), int_step=0.05)
    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")

    t_start = time.time()
    print("Plotting trajectories... ", end='')
    mp.plot_trajs(color_mode="reachability")
    x_dak, y_dak = 0, 0
    x_nat, y_nat = 1, 1
    mp.display.map.scatter(x_dak, y_dak, s=8., color='red', marker='D')
    mp.display.map.scatter(x_nat, y_nat, s=8., color='red', marker='D')
    mp.display.map.annotate('Dakar', (x_dak, y_dak), (0., -0.03))#, textcoords='offset points')
    mp.display.map.annotate('Natal', (x_nat, y_nat), xycoords='data')

    mp.display.map.set_ylim(-0.1, 1.1)
    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")
    plt.show()


if __name__ == '__main__':
    example3()
