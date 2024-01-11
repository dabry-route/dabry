from typing import Optional, Dict

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
                 events: Optional[Dict[str, ndarray]] = None,
                 info_dict: Optional[dict] = None):
        """
        :param times: Time stamps shape (n,)
        :param states: States (n, 2)
        :param coords: Type of coordinates : 'cartesian or 'gcs'
        :param controls: Controls (n-1, 2)
        :param costates: Costates (n, 2)
        :param cost: Instantaneous cost (n-1,)
        :param events: Dictionary of event time stamps following scipy's solve_ivp behavior
        :param info_dict: Additional information for the trajectory
        """
        self.times = times.copy()
        self.states = states.copy()
        self.controls = controls.copy() if controls is not None else None
        self.costates = costates.copy() if costates is not None else None
        self.cost = cost.copy() if cost is not None else None

        self.events = events

        self.coords = coords
        self.info_dict = info_dict

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
                   controls=controls, costates=costates, cost=cost, events=events, info_dict=info_dict)

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
                   controls=controls, costates=costates, cost=cost, events=events, info_dict=info_dict)
