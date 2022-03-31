import h5py
import matplotlib as mpl
import numpy as np

from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.shooting import Shooting
from mermoz.trajectory import dump_trajs
from mermoz.wind import UniformWind, DiscreteWind
from mermoz.misc import COORD_GCS, COORD_CARTESIAN

mpl.style.use('seaborn-notebook')


def example5():
    """
    Example of the shooting method on a vortex wind
    """
    # UAV airspeed in m/s
    v_a = 23.
    # The time window upper bound in seconds
    T = 10 * 3600. / 2000e3

    const_wind = UniformWind(np.array([0., 0.]))

    total_wind = DiscreteWind('/home/bastien/Documents/data/wind/mermoz/Dakar-Natal-0.5-tweaked/data2.h5')#  * (1 / 2000e3)

    coords = COORD_CARTESIAN

    x_van = -123.
    y_van = 49.
    x_hon = -157.
    y_hon = 21.

    # def domain(x):
    #     margin = 1e-1  # [rad]
    #     return -np.pi / 2. + margin < x[1] < np.pi / 2. - margin

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
    zermelo_model.update_wind(total_wind)


    # Initial point
    x_init = np.array([0.495, 0.57])
    print(zermelo_model.wind.value(np.array([0.6, 0.6])))
    # with h5py.File('/home/bastien/Documents/data/wind/mermoz/Dakar-Natal-0.5-tweaked/data.h5') as f:
    #     print(f['data'][:, :, 0])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map')

    # Set a list of initial adjoint states for the shooting method
    initial_headings = np.linspace(1e-3, 2 * np.pi - 1e-3, 50)
    list_p = list(map(lambda theta: -np.array([np.sin(theta), np.cos(theta)]), initial_headings))

    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined
    for p in list_p:
        shoot = Shooting(zermelo_model.dyn, x_init, T, N_iter=100, coords=coords)
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
    dump_trajs(mp.trajs, '/home/bastien/Documents/work/mermoz/output/trajs')
    # mp.plot_trajs(color_mode="reachability")
    # plt.show()


if __name__ == '__main__':
    example5()
