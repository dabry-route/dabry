import time

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from src.feedback import FixedHeadingFB
from src.mermoz import MermozProblem
from src.model import ZermeloGeneralModel
from src.shooting import Shooting
from src.stoppingcond import TimedSC
from wind import VortexWind, UniformWind

mpl.style.use('seaborn-notebook')


def example2():
    """
    Example of the shooting method on a vortex wind
    """
    print("Example 2")
    print("Building model... ", end='')
    t_start = time.time()
    # UAV airspeed in m/s
    v_a = 1.
    # UAV goal point x-coordinate in meters
    x_f = 1.
    # The time window upper bound in seconds
    T = 4.

    vortex1 = VortexWind(0.5, 0.7, -1.)
    vortex2 = VortexWind(0.8, 0.2, -0.5)
    vortex3 = VortexWind(0.6, -0.4, 0.8)
    const_wind = UniformWind(np.array([-0.1, 0.]))

    # Precise the domain for the state. Avoid the center of vortices.
    def domain(x):
        margin = 1e-1  # [m]
        b1 = np.linalg.norm(x - vortex1.omega) > margin
        b2 = np.linalg.norm(x - vortex2.omega) > margin
        b3 = np.linalg.norm(x - vortex3.omega) > margin
        return b1 and b2 and b3

        # Wind allows linear composition
    total_wind = 3. * const_wind + vortex1 + vortex2 + vortex3
    vortex1.value(np.array([0., 0.]))
    const_wind.value(np.array([0.3, 0.5]))
    total_wind.value(np.array([0.3, 0.5]))

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, x_f)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = np.array([0., 0.])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T)
    mp.display.set_wind_density(2)
    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")

    # Set a list of initial adjoint states for the shooting method
    initial_headings = np.linspace(- np.pi/4 + 1e-3, np.pi/4 - 1e-3, 20)
    list_p = list(map(lambda theta: -np.array([np.cos(theta), np.sin(theta)]), initial_headings))

    print(f"Shooting PMP trajectories ({len(list_p)})... ", end='')
    t_start = time.time()

    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined
    for k, p in enumerate(list_p):
        shoot = Shooting(zermelo_model.dyn, x_init, T, adapt_ts=True, N_iter=1000, ceil=3e-2, domain=domain, stop_on_failure=False)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)

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
    list_headings = np.linspace(-0.6, 1.2, 20)
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
    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")
    plt.show()


if __name__ == '__main__':
    example2()
