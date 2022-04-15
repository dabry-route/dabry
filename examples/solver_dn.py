import os
import random
import time

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize
from math import atan, cos, sin

from mermoz.feedback import FixedHeadingFB, TargetFB
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.solver import Solver
from mermoz.stoppingcond import TimedSC, DistanceSC
from mermoz.wind import VortexWind, UniformWind, LinearWind, DiscreteWind, RankineVortexWind
from mermoz.misc import *
from mdisplay.geodata import GeoData

mpl.style.use('seaborn-notebook')


def run():
    output_dir = '/home/bastien/Documents/work/mermoz/output/example_solver_dn'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)

    coords = COORD_GCS

    # UAV airspeed in m/s
    v_a = 23.

    gd = GeoData()
    # Initial point
    # x_init = DEG_TO_RAD * np.array((-60., 60.))
    x_init = np.array(gd.get_coords('Paris', units='rad'))

    # Target
    # x_target = DEG_TO_RAD * np.array((-38, 50.))
    x_target = np.array(gd.get_coords('New York', units='rad'))

    # Time window upper bound
    # Estimated through great circle distance + 20 percent
    T = 1.4 * geodesic_distance(x_init[0], x_init[1], x_target[0], x_target[1], mode='rad') / v_a

    total_wind = DiscreteWind()
    # total_wind.load('/home/bastien/Documents/data/wind/windy/NorthAtlantic.mz/data.h5')
    total_wind.load('/home/bastien/Documents/data/wind/windy/NewYork-Paris-1.0-padded.mz/data.h5')
    mdfm.dump_wind(total_wind)

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
    zermelo_model.update_wind(total_wind)

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, coords=coords, autodomain=True)

    # u_min = -3 * np.pi / 8.
    # u_max = 3 * np.pi / 8.
    # u_min = - np.pi / 8.
    # u_max = 0.
    # psi_min = DEG_TO_RAD * 50.
    # psi_max = DEG_TO_RAD * 100.
    auto_psi = atan((cos(x_target[1]) * sin(x_target[0] - x_init[0])) / (
                cos(x_init[1]) * sin(x_target[1]) - sin(x_init[1]) * cos(x_target[1]) * cos(x_target[0] - x_init[0])))
    print(f'auto_psi : {180/pi * auto_psi}')
    auto_psi = auto_psi + 2 * pi * (auto_psi < 0.)
    psi_min = auto_psi - DEG_TO_RAD * 20.
    psi_max = auto_psi + DEG_TO_RAD * 20.

    nt_pmp = 1000

    opti_ceil = EARTH_RADIUS / 100
    neighb_ceil = EARTH_RADIUS / 200

    solver = Solver(mp,
                    x_init,
                    x_target,
                    T,
                    psi_min,
                    psi_max,
                    output_dir,
                    N_disc_init=2,
                    opti_ceil=opti_ceil,
                    neighb_ceil=neighb_ceil,
                    n_min_opti=1,
                    adaptive_int_step=False,
                    N_iter=nt_pmp)

    solver.log_config()

    solver.setup()

    t_start = time.time()
    #solver.solve_fancy()
    t_end = time.time()
    time_pmp = t_end - t_start

    list_headings = DEG_TO_RAD * np.linspace(270, 280, 10)

    mp.load_feedback(TargetFB(mp.model.wind, v_a, x_target, mp.coords))
    mp.integrate_trajectory(x_init, DistanceSC(lambda x: geodesic_distance(x, x_target), opti_ceil), int_step=T / (nt_pmp - 1))

    mdfm.dump_trajs(mp.trajs)

    ps = ParamsSummary({}, output_dir)
    ps.load_from_solver(solver)
    ps.add_param('pmp_time', time_pmp)
    ps.dump()


if __name__ == '__main__':
    run()
