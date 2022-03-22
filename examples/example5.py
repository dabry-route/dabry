import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from src.feedback import FixedHeadingFB, WindAlignedFB
from src.mermoz import MermozProblem
from src.model import ZermeloGeneralModel
from src.shooting import Shooting
from src.stoppingcond import TimedSC
from src.wind import VortexWind, UniformWind
from src.misc import COORD_GCS

mpl.style.use('seaborn-notebook')


def example5():
    """
    Example of the shooting method on a vortex wind
    """
    # UAV airspeed in m/s
    v_a = 1.
    # The time window upper bound in seconds
    T = 2.

    const_wind = UniformWind(np.array([0., 0.]))

    # Wind allows linear composition
    total_wind = const_wind

    def domain(x):
        margin = 1e-6  # [rad]
        return -np.pi / 2. + margin < x[1] < np.pi / 2. - margin

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, 1., mode='plate-carree')
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = np.array([np.pi * -60 / 180, np.pi * 45 / 180])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map')

    # Set a list of initial adjoint states for the shooting method
    initial_headings = np.linspace(0.05, np.pi - 0.05, 20)
    list_p = list(map(lambda theta: -np.array([np.sin(theta), np.cos(theta)]), initial_headings))

    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined
    for p in list_p:
        shoot = Shooting(zermelo_model.dyn, x_init, T, N_iter=1000, domain=domain, coords=COORD_GCS)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)

    # Get also explicit control law traejctories
    # Here we add trajectories trying to steer the UAV
    # on a straight line starting from (0, 0) and with a given
    # heading angle
    '''
    list_headings = np.linspace(-0.6, 1.2, 10)
    for heading in list_headings:
        mp.load_feedback(FixedHeadingFB(mp._model.wind, v_a, heading))
        mp.integrate_trajectory(x_init, TimedSC(T), int_step=0.01)

    mp.load_feedback(WindAlignedFB(mp._model.wind))
    mp.integrate_trajectory(x_init, TimedSC(T), int_step=0.01)
    '''
    mp.plot_trajs(color_mode="reachability")
    plt.show()


if __name__ == '__main__':
    example5()
