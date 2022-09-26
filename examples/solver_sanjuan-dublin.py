import os
import random
import time

import matplotlib as mpl
import numpy as np
from math import atan, cos, sin, atan2
from pyproj import Proj

from mermoz.feedback import FixedHeadingFB, GSTargetFB, ConstantFB
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.solver import Solver
from mermoz.stoppingcond import TimedSC, DistanceSC
from mermoz.wind import VortexWind, UniformWind, LinearWind, DiscreteWind, RankineVortexWind
from mermoz.misc import *
from mermoz.rft import RFT
from mdisplay.geodata import GeoData

mpl.style.use('seaborn-notebook')


def run():
    flat = True
    if flat:
        output_dir = '/home/bastien/Documents/work/mermoz/output/example_solver_san-juan_dublin_ortho'
    else:
        output_dir = '/home/bastien/Documents/work/mermoz/output/example_solver_san-juan_dublin'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    mdfm.set_output_dir(output_dir)

    if flat:
        coords = COORD_CARTESIAN
    else:
        coords = COORD_GCS

    # UAV airspeed in m/s
    v_a = 23.

    # Wind
    total_wind = DiscreteWind(force_analytical=True, interp='linear')
    # total_wind.load('/home/bastien/Documents/data/wind/windy/NorthAtlantic.mz/data.h5')
    if flat:
        total_wind.load('/home/bastien/Documents/data/wind/ncdc/san-juan-dublin-flattened-ortho.mz/wind.h5')
    else:
        total_wind.load('/home/bastien/Documents/work/mermoz/output/example_wf_san-juan_dublin/wind.h5')
    mdfm.dump_wind(total_wind)

    gd = GeoData()
    # Initial point

    if flat:
        proj = Proj(proj='ortho', lon_0=total_wind.lon_0, lat_0=total_wind.lat_0)

        point = np.array(gd.get_coords('San Juan', units='deg'))
        x_init = np.array(proj(*point))

        point = np.array(gd.get_coords('Dublin', units='deg'))
        x_target = np.array(proj(*point))
    else:
        x_init = np.array(gd.get_coords('San Juan', units='rad'))
        x_target = np.array(gd.get_coords('Dublin', units='rad'))

    # Time window upper bound
    # Estimated through great circle distance + 20 percent
    if flat:
        T = 1.1 * np.linalg.norm(x_target - x_init) / v_a
    else:
        T = 1.1 * geodesic_distance(x_init[0], x_init[1], x_target[0], x_target[1], mode='rad') / v_a

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
    zermelo_model.update_wind(total_wind)

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, coords=coords, autodomain=True, mask_land=False)

    if flat:
        auto_psi = atan2(x_target[1] - x_init[1], x_target[0] - x_init[0])
    else:
        auto_psi = atan((cos(x_target[1]) * sin(x_target[0] - x_init[0])) / (
                cos(x_init[1]) * sin(x_target[1]) - sin(x_init[1]) * cos(x_target[1]) * cos(x_target[0] - x_init[0])))
        print(f'auto_psi : {180 / pi * auto_psi}')
        auto_psi = auto_psi + 2 * pi * (auto_psi < 0.)
        auto_psi += 0.
    psi_min = auto_psi - DEG_TO_RAD * 5.
    psi_max = auto_psi + DEG_TO_RAD * 5.

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
                    N_disc_init=10,
                    opti_ceil=opti_ceil,
                    neighb_ceil=neighb_ceil,
                    n_min_opti=1,
                    adaptive_int_step=False,
                    nt_pmp=nt_pmp)

    solver.log_config()

    solver.setup()

    t_start = time.time()
    solver.solve_fancy()
    t_end = time.time()
    time_pmp = t_end - t_start

    if flat:

        nx_rft = 201
        ny_rft = 201
        nt_rft = 30
        bl = np.array((total_wind.x_min, total_wind.y_min))
        tr = np.array((total_wind.x_max, total_wind.y_max))

        print(f"Tracking reachability front ({nx_rft}x{ny_rft})... ", end='')
        t_start = time.time()

        rft = RFT(bl, tr, T, nx_rft, ny_rft, nt_rft, mp, x_init, kernel='matlab', coords=COORD_CARTESIAN)

        rft.compute()

        t_end = time.time()
        time_rft = t_end - t_start
        print(f"Done ({time_rft:.3f} s)")

        rft.dump_rff(output_dir)

    mp.load_feedback(GSTargetFB(mp.model.wind, v_a, x_target, mp.coords))
    mp.integrate_trajectory(x_init, DistanceSC(lambda x: distance(x, x_target, coords=coords), opti_ceil), int_step=T / (nt_pmp - 1))

    mdfm.dump_trajs(mp.trajs)

    ps = ParamsSummary({}, output_dir)
    ps.load_from_solver(solver)
    ps.add_param('pmp_time', time_pmp)
    ps.dump()


if __name__ == '__main__':
    run()
