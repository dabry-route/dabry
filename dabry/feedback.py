from abc import ABC, abstractmethod

import numpy as np
from numpy import arctan2 as atan2
from numpy import ndarray
from numpy import sin, cos

from dabry.misc import Utils, directional_timeopt_control, Coords
from dabry.flowfield import FlowField

"""
feedback.py
Feedback control laws for vehicles in flow fields controlled both
in heading and airspeed.

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


class Feedback(ABC):
    """
    Defines feedback laws for given models
    """

    @abstractmethod
    def __call__(self, t: float, x: ndarray) -> ndarray:
        """
        :param t: The time at which to compute the control
        :param x: The state
        :return: The feedback
        """
        pass


class ConstantFB(Feedback):

    def __init__(self, value: ndarray):
        self.value = value.copy()

    def __call__(self, t, x):
        return self.value


class GSTargetFB(Feedback):
    """
    Control law trying to put ground speed vector towards a fixed target
    """

    def __init__(self, ff: FlowField, srf: float, target: ndarray):
        self.ff = ff
        self.srf = srf
        self.target = target.copy()

    def __call__(self, t, x):
        if self.ff.coords == Coords.GCS:
            # Got to 3D cartesian assuming spherical earth
            lon, lat = x[0], x[1]
            X3 = Utils.EARTH_RADIUS * np.array((cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)))
            # Vector normal to earth at position
            e_phi = np.array((-sin(lon), cos(lon), 0.))
            e_lambda = np.array((-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)))
            lon, lat = self.target[0], self.target[1]
            X_target3 = Utils.EARTH_RADIUS * np.array((cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)))
            e_target = np.zeros(2)
            e_target[0] = (X_target3 - X3) @ e_phi
            e_target[1] = (X_target3 - X3) @ e_lambda
        else:
            #  self.coords == COORD_CARTESIAN
            e_target = np.zeros(2)
            e_target[:] = self.target - x

        if np.isclose(np.linalg.norm(e_target), 0):
            return np.zeros(2)

        return directional_timeopt_control(self.ff.value(t, x), e_target, self.srf)


class HTargetFB(Feedback):
    """
    Control law heading towards target
    """

    def __init__(self, target: ndarray, coords: Coords):
        self.target = target.copy()
        self.coords = coords

    def __call__(self, t, x):
        if self.coords == Coords.CARTESIAN:
            e_target = np.zeros(2)
            e_target[:] = self.target - x

            if np.isclose(np.linalg.norm(e_target), 0):
                return np.zeros(2)

            e_target = e_target / np.linalg.norm(e_target)
            res = atan2(e_target[1], e_target[0])
            return np.array((np.cos(res), np.sin(res)))
        else:
            # Got to 3D cartesian assuming spherical earth
            lon, lat = x[0], x[1]
            # Vector normal to earth at position
            X3 = Utils.EARTH_RADIUS * np.array((cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)))
            e_phi = np.array((-sin(lon), cos(lon), 0.))
            e_lambda = np.array((-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)))
            lon, lat = self.target[0], self.target[1]
            X_target3 = Utils.EARTH_RADIUS * np.array((cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)))
            e_target = np.zeros(2)
            e_target[0] = (X_target3 - X3) @ e_phi
            e_target[1] = (X_target3 - X3) @ e_lambda

            if np.isclose(np.linalg.norm(e_target), 0):
                return np.zeros(2)

            e_target = e_target / np.linalg.norm(e_target)
            res = atan2(e_target[1], e_target[0])
            return np.array((np.cos(res), np.sin(res)))


class MapFB(Feedback):

    def __init__(self, grid, values):
        """
        :param grid: Grid of points for discretization (nx, ny, 2)
        :param values: Heading values on the grid (nx - 1, ny - 1)
        """
        self.grid = np.array(grid)
        self.values = np.array(values)

    def __call__(self, _, x):
        xx = (x[0] - self.grid[0, 0, 0]) / (self.grid[-1, 0, 0] - self.grid[0, 0, 0])
        yy = (x[1] - self.grid[0, 0, 1]) / (self.grid[0, -1, 1] - self.grid[0, 0, 1])

        nx, ny, _ = self.grid.shape
        i = int((nx - 1) * xx)
        j = int((ny - 1) * yy)

        return self.values[i, j]
