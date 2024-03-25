import json
import os
import warnings
from typing import Optional, Dict

import numpy as np
from numpy import ndarray

from dabry.misc import Coords

"""
trajectory.py
Handling trajectory data structure.

Copyright (C) 2021 Bastien Schnitzler 
(bastien dot schnitzler at live dot fr)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


def traj_name_to_filename(name, meta=False):
    suffix = '.npz' if not meta else '_meta.json'
    return ("traj_%s" % name) + suffix


def traj_filepath_to_name(filepath):
    filename = os.path.basename(filepath)
    if not filename.startswith('traj_'):
        raise ValueError('Filename "%s" is not properly formatted' % filename)
    return filename[5:].split('.')[0]


class Trajectory:
    """
    The definition of a trajectory for the navigation problem
    """

    def __init__(self,
                 times: ndarray,
                 states: ndarray,
                 controls: Optional[ndarray] = None,
                 costates: Optional[ndarray] = None,
                 cost: Optional[ndarray] = None,
                 events: Optional[Dict[str, ndarray]] = None):
        """
        :param times: Time stamps shape (n,)
        :param states: States (n, 2)
        :param controls: Controls (n-1, 2)
        :param costates: Costates (n, 2)
        :param cost: Running (integrated) cost (n-1,)
        :param events: Dictionary of event time stamps following scipy's solve_ivp behavior
        """
        self.times: ndarray = times.copy()
        self.states = states.copy()
        self.controls = controls.copy() if controls is not None else None
        self.costates = costates.copy() if costates is not None else None
        self.cost = cost.copy() if cost is not None else None

        self.events: Dict[str, ndarray] = events if events is not None else {}

    @classmethod
    def empty(cls):
        return cls(np.array(()), np.array(((), ())), np.array(((), ())), np.array(((), ())), np.array(()))

    def __add__(self, other):
        if not isinstance(other, Trajectory):
            raise ValueError('Trajectory addition with type %s' % type(other))
        if len(self) == 0:
            times = other.times
            states = other.states
            controls = other.controls
            costates = other.costates
            cost = other.cost
        elif len(other) == 0:
            times = self.times
            states = self.states
            controls = self.controls
            costates = self.costates
            cost = self.cost
        else:
            times = np.concatenate((self.times, other.times))
            states = np.concatenate((self.states, other.states))
            controls = np.concatenate(
                ((self.controls if self.controls is not None else np.ones((self.times.shape[0], 2)) * np.nan),
                 (other.controls if other.controls is not None else np.ones((other.times.shape[0], 2)) * np.nan))
            )
            costates = np.concatenate(
                ((self.costates if self.costates is not None else np.ones((self.times.shape[0], 2)) * np.nan),
                 (other.costates if other.costates is not None else np.ones((other.times.shape[0], 2)) * np.nan))
            )

            cost = np.concatenate((self.cost if self.cost is not None else np.ones(self.times.shape[0]) * np.nan,
                                   other.cost if other.cost is not None else np.ones(other.times.shape[0]) * np.nan))

        events = self.events.copy()
        events.update(other.events)

        return Trajectory(times, states, controls=controls, costates=costates, cost=cost, events=events)

    def __len__(self):
        return self.times.shape[0]

    def copy(self):
        return Trajectory(self.times.copy(), self.states.copy(), controls=self.controls.copy(),
                          costates=self.costates.copy(), cost=self.cost.copy(), events=self.events.copy())

    def save(self, name, dir_name, scale_length: Optional[float] = None, scale_time: Optional[float] = None,
             bl: Optional[ndarray] = None, time_offset: Optional[float] = None):
        scale_length = 1 if scale_length is None else scale_length
        scale_time = 1 if scale_time is None else scale_time
        time_offset = 0 if time_offset is None else time_offset
        bl = np.array((0, 0)) if bl is None else bl.copy()
        np.savez(os.path.join(dir_name, traj_name_to_filename(name)),
                 times=time_offset + self.times * scale_time, states=bl + self.states * scale_length,
                 costates=self.costates if self.costates is not None else np.array(((), ())),
                 controls=self.controls if self.controls is not None else np.array(((), ())),
                 cost=self.cost * scale_time if self.cost is not None else np.array(()))
        meta_data = {'events': {}}
        for e_name, times in self.events.items():
            meta_data['events'][e_name] = times.tolist()
        meta_fpath = os.path.join(dir_name, traj_name_to_filename(name, meta=True))
        with open(meta_fpath, 'w') as f:
            json.dump(meta_data, f)

    @classmethod
    def from_npz(cls, filepath):
        """
        :param filepath: Should end with ".npz"
        :return:
        """
        if not filepath.endswith('.npz'):
            raise ValueError('Not a NPZ file %s' % filepath)
        data = np.load(filepath)
        try:
            meta_data = json.load(open(filepath[:-4] + '_meta.json'))
            events = {}
            for k, v in meta_data['events'].items():
                events[k] = np.array(v)
        except FileNotFoundError:
            warnings.warn('Metadata not found for trajectory', category=UserWarning)
            events = {}

        controls = data['controls'] if data['controls'].size > 0 else None
        costates = data['costates'] if data['costates'].size > 0 else None
        cost = data['cost'] if data['cost'].size > 0 else None
        return cls(data['times'], data['states'], controls, costates, cost, events)
