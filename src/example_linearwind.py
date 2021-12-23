import matplotlib as mpl
import numpy as np

from src.feedback import FixedHeadingFB, WindAlignedFB
from src.mermoz import MermozProblem
from src.model import ZermeloGeneralModel
from src.shooting import Shooting
from src.stoppingcond import TimedSC
from wind import VortexWind, UniformWind, LinearWind

mpl.style.use('seaborn-notebook')


def example_linear_wind():
    """
    Example of the shooting method on a vortex wind
    """
    # UAV airspeed in m/s
    v_a = 1.
    # UAV goal point x-coordinate in meters
    x_f = 1.
    # The time window upper bound in seconds
    T = 2.
    # The wind gradient
    gradient = np.array([[0., v_a/10.],
                         [0., 0.]])
    origin = np.array([0., 0.])
    value_origin = np.array([0., 0.])

    linear_wind = LinearWind(gradient, origin, value_origin)

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, x_f)
    zermelo_model.update_wind(linear_wind)

    # Initial point
    x_init = np.array([0., 0.])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map')

    # Set a list of initial adjoint states for the shooting method
    initial_headings = np.linspace(- np.pi/16. + 1e-3, np.pi/16. - 1e-3, 30)
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
        mp.load_feedback(FixedHeadingFB(mp._model.wind, v_a, heading))
        mp.integrate_trajectory(x_init, TimedSC(T), int_step=0.01)

    mp.load_feedback(WindAlignedFB(mp._model.wind))
    mp.integrate_trajectory(x_init, TimedSC(T), int_step=0.01)

    mp.eliminate_trajs(1e-1)

    mp.plot_trajs(color_mode="reachability")


if __name__ == '__main__':
    example_linear_wind()
