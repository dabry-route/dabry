import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from mermoz.feedback import FixedHeadingFB, WindAlignedFB
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.shooting import Shooting
from mermoz.stoppingcond import TimedSC
from mermoz.wind import VortexWind, UniformWind

mpl.style.use('seaborn-notebook')


def example1():
    """
    Example of the shooting method on a vortex wind
    """
    # UAV airspeed in m/s
    v_a = 1.
    # UAV goal point x-coordinate in meters
    x_f = 1.
    # The time window upper bound in seconds
    T = 3.
    # The vortex center definition in meters
    omega = np.array([0.5, 0.3])
    # The vortex intensity in m^2/s
    gamma = -1.5

    vortex_wind = VortexWind(omega[0], omega[1], gamma)
    const_wind = UniformWind(np.array([-0.1, 0.]))

    # Wind allows linear composition
    total_wind = 3. * const_wind + vortex_wind

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = np.array([0., 0.])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='full-adjoint')

    # Set a list of initial adjoint states for the shooting method
    initial_headings = np.linspace(- np.pi + 1e-3, np.pi - 1e-3, 200)
    list_p = list(map(lambda theta: -np.array([np.cos(theta), np.sin(theta)]), initial_headings))

    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined
    for p in list_p:
        shoot = Shooting(zermelo_model.dyn, x_init, T, N_iter=100)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)

    # Get also explicit control law traejctories
    # Here we add trajectories trying to steer the UAV
    # on a straight line starting from (0, 0) and with a given
    # heading angle
    list_headings = np.linspace(-0.6, 1.2, 10)
    for heading in list_headings:
        mp.load_feedback(FixedHeadingFB(mp.model.wind, v_a, heading))
        mp.integrate_trajectory(x_init, TimedSC(T), int_step=0.01)

    mp.load_feedback(WindAlignedFB(mp.model.wind))
    mp.integrate_trajectory(x_init, TimedSC(T), int_step=0.01)

    mp.plot_trajs(color_mode="reachability")
    plt.show()


if __name__ == '__main__':
    example1()
