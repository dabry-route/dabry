import os
import matplotlib as mpl
import numpy as np

from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.shooting import Shooting
from mermoz.trajectory import dump_trajs
from mermoz.wind import UniformWind, RealWind
from mermoz.misc import COORD_GCS, COORD_CARTESIAN

mpl.style.use('seaborn-notebook')

deg_to_rad = np.pi / 180.


def example_geodesic():
    """
    Example of geodesic approximation with extremals
    """
    # UAV airspeed in m/s
    v_a = 23.
    # The time window upper bound in seconds
    T = 70 * 3600.

    # Set no wind for this case
    total_wind = UniformWind(np.array([0., 0.]))

    # GCS mode
    coords = COORD_GCS

    lon_ny = -73.935242
    lat_ny = 40.730610
    lon_par = 2.349014
    lat_par = 48.864716

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
    zermelo_model.update_wind(total_wind)

    # Initial point at NY
    x_init = deg_to_rad * np.array([lon_ny, lat_ny])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map')

    # Set a list of initial heading values for the shooting method
    initial_headings = np.linspace(np.pi/4., 2*np.pi/7., 10)
    list_p = list(map(lambda theta: -np.array([np.sin(theta), np.cos(theta)]), initial_headings))

    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined
    for p in list_p:
        shoot = Shooting(zermelo_model.dyn, x_init, T, N_iter=1000, coords=coords)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)

    output_path = '../output/example_geodesic'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    dump_trajs(mp.trajs, output_path)


if __name__ == '__main__':
    example_geodesic()
