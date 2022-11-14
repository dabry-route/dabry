import random
import time
import os

import h5py

from mermoz.misc import *
from mermoz.trajectory import Trajectory
from mermoz.post_processing import PostProcessing
from mermoz.mdf_manager import MDFmanager
from mermoz.params_summary import ParamsSummary
from mermoz.solver_rp import SolverRP

def get_latest_output_dir(base_dir):
    nlist = [dd for dd in os.listdir(base_dir) if
             os.path.isdir(os.path.join(base_dir, dd)) and not dd.startswith('.')]
    latest_subdir = max(nlist, key=lambda name: os.path.getmtime(os.path.join(base_dir, name)))
    return os.path.join(base_dir, latest_subdir)

if __name__ == '__main__':

    base_dir = f'/home/bastien/Documents/work/mermoz/output/'
    output_dir = get_latest_output_dir(base_dir)
    # output_dir = f'/home/bastien/Documents/work/mermoz/output/example_energy_band'
    print(output_dir)
    pp = PostProcessing(output_dir)
    pp.load()

    pp.stats()
