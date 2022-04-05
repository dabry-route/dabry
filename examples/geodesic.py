import os
import matplotlib as mpl
import numpy as np

from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.shooting import Shooting
from mermoz.wind import UniformWind, DiscreteWind
from mermoz.misc import *

from mdisplay.geodata import GeoData

mpl.style.use('seaborn-notebook')


def run():
    """
    Example of geodesic approximation with extremals
    """
    output_dir = '../output/example_geodesic'
    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)

    # UAV airspeed in m/s
    v_a = 23.
    # The time window upper bound in seconds
    T = 71 * 3600.

    # Set no wind for this case
    total_wind = UniformWind(np.array([0., 0.]))

    # GCS mode
    coords = COORD_GCS

    gd = GeoData()
    coords_ny = gd.get_coords('new york')

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
    zermelo_model.update_wind(total_wind)

    # Initial point at NY
    x_init = DEG_TO_RAD * np.array(coords_ny)

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map')

    # Set a list of initial heading values for the shooting method
    initial_headings = np.linspace(np.pi / 4., 2 * np.pi / 7., 10)
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

    mdfm.dump_trajs(mp.trajs)

    params = {
        'coords': 'cartesian',
        'point_init': (x_init[0], x_init[1]),
        'max_time': T,
    }

    ps = ParamsSummary(params, output_dir)
    ps.dump()


if __name__ == '__main__':
    run()
