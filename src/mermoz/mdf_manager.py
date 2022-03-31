import os
import sys

import h5py
import numpy as np

from mermoz.misc import *
from mermoz.wind import Wind


class MDFmanager:
    """
    This class handles the writing and reading of Mermoz Data Format (MDF) files
    """

    def __init__(self):
        self.output_dir = None
        self.trajs_filename = 'trajectories.h5'
        self.wind_filename = 'wind.h5'

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def dump_trajs(self, traj_list, filename=None):
        filename = self.trajs_filename if filename is None else filename
        filepath = os.path.join(self.output_dir, filename)
        with h5py.File(filepath, "w") as f:
            for i, traj in enumerate(traj_list):
                nt = traj.timestamps.shape[0]
                trajgroup = f.create_group(str(i))
                trajgroup.attrs['type'] = traj.type
                trajgroup.attrs['coords'] = traj.coords
                trajgroup.attrs['interrupted'] = traj.interrupted
                trajgroup.attrs['last_index'] = traj.last_index
                trajgroup.attrs['label'] = traj.label

                factor = (180 / np.pi if traj.coords == COORD_GCS else 1.)

                dset = trajgroup.create_dataset('data', (nt, 2), dtype='f8')
                dset[:, :] = traj.points * factor

                dset = trajgroup.create_dataset('ts', (nt,), dtype='f8')
                dset[:] = traj.timestamps

                dset = trajgroup.create_dataset('controls', (nt,), dtype='f8')
                dset[:] = traj.controls

                if hasattr(traj, 'adjoints'):
                    dset = trajgroup.create_dataset('adjoints', (nt, 2), dtype='f8')
                    dset[:, :] = traj.adjoints

    def print_trajs(self, filepath):
        with h5py.File(filepath, 'r') as f:
            for traj in f.values():
                print(traj)
                for attr, val in traj.attrs.items():
                    print(f'{attr} : {val}')

    def dump_wind(self, wind: Wind, filename=None):
        if not wind.is_dumpable:
            print('Error : Wind is not dumpable to file', file=sys.stderr)
            exit(1)
        filename = self.wind_filename if filename is None else filename
        filepath = os.path.join(self.output_dir, filename)
        with h5py.File(filepath, 'w') as f:
            f.attrs['coords'] = wind.coords
            f.attrs['units_grid'] = U_DEG
            dset = f.create_dataset('data', (wind.nt, wind.nx, wind.ny, 2), dtype='f8')
            dset[:] = wind.uv
            dset = f.create_dataset('ts', (wind.nt,), dtype='f8')
            dset[:] = wind.ts
            factor = (1 / DEG_TO_RAD if wind.coords == COORD_GCS else 1.)
            dset = f.create_dataset('grid', (wind.nx, wind.ny, 2), dtype='f8')
            dset[:] = factor * wind.grid


if __name__ == '__main__':
    mdfm = MDFmanager()
    mdfm.print_trajs('/home/bastien/Documents/work/mermoz/output/example_front_tracking2/trajectories.h5')
