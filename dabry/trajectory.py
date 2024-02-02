import json
import warnings
from typing import Optional, Dict

import numpy as np
from numpy import ndarray

from dabry.misc import Utils

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


class Trajectory:
    """
    The definition of a trajectory for the navigation problem
    """

    def __init__(self,
                 times: ndarray,
                 states: ndarray,
                 coords,
                 controls: Optional[ndarray] = None,
                 costates: Optional[ndarray] = None,
                 cost: Optional[ndarray] = None,
                 events: Optional[Dict[str, ndarray]] = None):
        """
        :param times: Time stamps shape (n,)
        :param states: States (n, 2)
        :param coords: Type of coordinates : 'cartesian or 'gcs'
        :param controls: Controls (n-1, 2)
        :param costates: Costates (n, 2)
        :param cost: Instantaneous cost (n-1,)
        :param events: Dictionary of event time stamps following scipy's solve_ivp behavior
        """
        self.times: ndarray = times.copy()
        self.states = states.copy()
        self.controls = controls.copy() if controls is not None else None
        self.costates = costates.copy() if costates is not None else None
        self.cost = cost.copy() if cost is not None else None

        self.events: Dict[str, ndarray] = events if events is not None else {}

        self.coords = coords

    @classmethod
    def cartesian(cls,
                  times: ndarray,
                  states: ndarray,
                  controls: Optional[ndarray] = None,
                  costates: Optional[ndarray] = None,
                  cost: Optional[ndarray] = None,
                  events: Optional[Dict[str, ndarray]] = None,
                  info_dict: Optional[dict] = None):
        return cls(times, states, Utils.COORD_CARTESIAN,
                   controls=controls, costates=costates, cost=cost, events=events)

    @classmethod
    def gcs(cls,
            times: ndarray,
            states: ndarray,
            controls: Optional[ndarray] = None,
            costates: Optional[ndarray] = None,
            cost: Optional[ndarray] = None,
            events: Optional[Dict[str, ndarray]] = None,
            info_dict: Optional[dict] = None):
        return cls(times, states, Utils.COORD_GCS,
                   controls=controls, costates=costates, cost=cost, events=events)

    def __add__(self, other):
        if not isinstance(other, Trajectory):
            raise ValueError('Trajectory addition with type %s' % type(other))
        if self.coords != other.coords:
            raise ValueError('Incompatible coord types %s and %s' % (self.coords, other.coords))
        if self.times.shape[0] == 0:
            return other
        if other.times.shape[0] == 0:
            return self
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

        events = self.events.copy().update(other.events)

        return Trajectory(times, states, self.coords,
                          controls=controls, costates=costates, cost=cost, events=events)

    # def fill_with(self, other):
    #     if not isinstance(other, Trajectory):
    #         raise ValueError('%s is not a Trajectory' % type(other))
    #     for values, values_other in zip([self.times, self.states, self.controls, self.costates, self.cost],
    #         [other.times, other.states, other.controls, other.costates, other.cost]):
    #         if values is None:
    #             continue
    #         if np.any(np.logical_and(np.logical_not(np.isnan(values)), np.logical_not(np.isnan(values_other)))):
    #             raise ValueError("Trajectories share non-nan values at similar positions")
    #     slc = np.logical_not(np.isnan(other.times))
    #     self.times[slc] = other.times[slc]
    #     self.states[slc] = other.states[slc]
    #     self.controls[slc] = other.controls[slc]
    #     self.costates[slc] = other.costates[slc]
    #     self.cost[slc] = other.cost[slc]

    def __len__(self):
        return self.times.shape[0]

    def copy(self):
        return Trajectory(self.times.copy(), self.states.copy(), self.coords, controls=self.controls.copy(),
                          costates=self.costates.copy(), cost=self.cost.copy(), events=self.events.copy())

    def save(self, filepath):
        np.savez(filepath, times=self.times, states=self.states,
                 costates=self.costates if self.costates is not None else np.array(()),
                 controls=self.controls if self.controls is not None else np.array(()),
                 cost=self.cost if self.cost is not None else np.array(()))
        meta_data = {'coords': self.coords, 'events': {}}
        for e_name, times in self.events.items():
            meta_data['events'][e_name] = times.tolist()
        meta_fpath = filepath + '_meta.json'
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
            coords = meta_data['coords']
            events = {}
            for k, v in meta_data['events'].items():
                events[k] = np.array(v)
        except FileNotFoundError:
            warnings.warn('Metadata not found for trajectory', category=UserWarning)
            coords = Utils.COORD_CARTESIAN
            events = {}

        controls = data['controls'] if data['controls'].size > 0 else None
        costates = data['costates'] if data['costates'].size > 0 else None
        cost = data['cost'] if data['cost'].size > 0 else None
        return cls(data['times'], data['states'], coords, controls, costates, cost, events)
