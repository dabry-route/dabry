import os
import time
from math import atan2

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from tqdm import tqdm

from mermoz.feedback import FunFB, ConstantFB, GSTargetFB, GreatCircleFB, HTargetFB, FixedHeadingFB, FunAS, E_GSTargetFB
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem, DatabaseProblem
from mermoz.shooting import Shooting
from mermoz.solver_ef import SolverEF
from mermoz.trajectory import Trajectory
from mermoz.solver_rp import SolverRP
from mermoz.stoppingcond import TimedSC, DistanceSC, DisjunctionSC
from mermoz.wind import DiscreteWind

if __name__ == '__main__':
    # Choose problem ID
    pb_id, seed = 10, 0
    dbpb = '72W_15S_0W_57S_20220301_12' #'37W_8S_16W_17S_20220301_12'

    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = MDFmanager()
    if dbpb is not None:
        output_dir = f'/home/bastien/Documents/work/mermoz/output/example_wind_{dbpb}'
    else:
        output_dir = f'/home/bastien/Documents/work/mermoz/output/example_wind_{IndexedProblem.problems[pb_id][1]}'
    mdfm.set_output_dir(output_dir)
    mdfm.clean_output_dir()

    nx_rft = 101
    ny_rft = 101
    nt_rft = 50

    # Create problem
    # mdfm.dump_wind_from_grib2(grib_fps, bl, tr)
    # pb = DatabaseProblem('/home/bastien/Documents/data/wind/ncdc/tmp.mz/wind.h5')
    if dbpb is not None:
        pb = DatabaseProblem(os.path.join('/home/bastien/Documents/data/wind/ncdc/', dbpb, 'wind.h5'), airspeed=23.)
    else:
        pb = IndexedProblem(pb_id, seed=seed)

    chrono.start('Dumping windfield to file')
    mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
    chrono.stop()