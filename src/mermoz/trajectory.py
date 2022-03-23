import numpy as np
from numpy import ndarray
import h5py
import os
from .misc import TRAJ_INT, COORD_GCS, COORD_CARTESIAN


def dump_trajs(traj_list, filepath):
    with h5py.File(os.path.join(filepath, 'trajectories.h5'), "w") as f:
        for i, traj in enumerate(traj_list):
            nt = traj.timestamps.shape[0]
            trajgroup = f.create_group(str(i))
            trajgroup.attrs['type'] = traj.type
            trajgroup.attrs['coords'] = traj.coords
            trajgroup.attrs['interrupted'] = traj.interrupted
            trajgroup.attrs['last_index'] = traj.last_index

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


class Trajectory:
    """
    The definition of a trajectory for the Mermoz problem
    """

    def __init__(self,
                 timestamps: ndarray,
                 points: ndarray,
                 controls: ndarray,
                 last_index: int,
                 optimal=False,
                 interrupted=False,
                 type=TRAJ_INT,
                 coords=COORD_GCS):
        """
        :param timestamps: A list of timestamps (t_0, ..., t_N) at which the following values were computed
        :param points: A list of points ((x_0, y_0), ..., (x_N, y_N)) describing the trajectory
        :param controls: A list of controls (u_0, ..., u_N) applied to follow the previous trajectory
        :param last_index: The index of the last significant value
        :param optimal: Indicates if trajectory is optimal or not (for plotting)
        :param interrupted: Indicates if trajectory was interrupted during construction
        :param type: Gives the type of the trajectory : 'integral' or 'pmp'
        :param coords: Type of coordinates : 'cartesian or 'gcs'
        """
        self.timestamps = np.zeros(timestamps.shape)
        self.timestamps[:] = timestamps

        self.points = np.zeros(points.shape)
        self.points[:] = points

        self.controls = np.zeros(controls.shape)
        self.controls[:] = controls

        self.last_index = last_index
        self.interrupted = interrupted
        self.optimal = optimal
        self.type = type
        self.coords = coords

    def get_final_time(self):
        return self.timestamps[self.last_index]

    def get_dt_stats(self):
        dt_list = np.zeros(self.timestamps.size - 1)
        dt_list[:] = self.timestamps[1:] - self.timestamps[:-1]
        return np.min(dt_list), np.max(dt_list), np.average(dt_list)


class AugmentedTraj(Trajectory):
    """
    Trajectory augmented by adjoint state
    """

    def __init__(self,
                 timestamps: ndarray,
                 points: ndarray,
                 adjoints: ndarray,
                 controls: ndarray,
                 last_index: int,
                 optimal=False,
                 interrupted=False,
                 type=TRAJ_INT,
                 coords=COORD_GCS):
        """
        :param adjoints: A list of adjoint states ((p_x0, p_y0), ..., (p_xN, p_yN))
        """
        super().__init__(timestamps, points, controls, last_index, optimal, interrupted, type, coords)
        self.adjoints = np.zeros(adjoints.shape)
        self.adjoints[:] = adjoints
