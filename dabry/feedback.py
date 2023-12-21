from abc import ABC, abstractmethod

import numpy as np
from numpy import arctan2 as atan2
from numpy import ndarray
from numpy import sin, cos

from dabry.misc import Utils
from dabry.wind import Wind, UniformWind

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
    def value(self, t: float, x: ndarray) -> ndarray:
        """
        :param t: The time at which to compute the control
        :param x: The state
        :return: The feedback
        """
        pass


class ConstantFB(Feedback):

    def __init__(self, value: ndarray):
        self.value = value.copy()

    def value(self, t, x):
        return self.value


class FixedHeadingFB(Feedback):
    """
    Defines a control law to follow a straight line from initial position
    defined by its angle to the x-axis.
    When the flow field value is too high for the vehicle to keep its line, the
    control law steers perpendicular to the target direction
    """

    def __init__(self, wind: Wind, srf: float, initial_steering: float):
        """
        :param wind: Wind object
        :param srf: Vehicle's speed relative to flow field
        """
        self.wind = wind
        self.srf = srf
        self.theta_0 = initial_steering

    def value(self, t, x):
        if self.wind.coords == Utils.COORD_CARTESIAN:
            e_theta_0 = np.array([np.cos(self.theta_0), np.sin(self.theta_0)])
        else:
            # coords gcs
            e_theta_0 = np.array([np.sin(self.theta_0), np.cos(self.theta_0)])
        wind = self.wind.value(t, x)
        wind_ortho = np.cross(e_theta_0, wind)
        r = -wind_ortho / self.srf
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = np.arcsin(r)
            if self.wind.coords == Utils.COORD_GCS:
                res *= -1
        res += self.theta_0
        return np.array((np.cos(res), np.sin(res)))


class GreatCircleFB(Feedback):
    """
    Control law for GCS problems only.
    Tries to stay on a great circle when wind allows it.
    """

    def __init__(self, wind: Wind, srf: float, target: ndarray):
        self.wind = wind
        self.srf = srf
        self.lon_t, self.lat_t = target[0], target[1]

    def value(self, t, x):
        # First get the desired heading to follow great circle
        lon, lat = x
        lon_t, lat_t = self.lon_t, self.lat_t
        u0 = np.arctan(1. / (np.cos(lat) * np.tan(lat_t) / np.sin(lon_t - lon) - np.sin(lat) / np.tan(lon_t - lon)))

        e_0 = np.array([np.sin(u0), np.cos(u0)])
        wind = self.wind.value(t, x)
        wind_ortho = np.cross(e_0, wind)
        r = -wind_ortho / self.srf
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = -np.arcsin(r)
        res += u0
        return np.array((np.cos(res), np.sin(res)))


class GSTargetFB(Feedback):
    """
    Control law trying to put ground speed vector towards a fixed target
    """

    def __init__(self, wind: Wind, srf: float, target: ndarray):
        self.wind = wind
        self.srf = srf
        self.target = target.copy()
        self.zero_ceil = 1e-3

    def value(self, t, x):
        if self.wind.coords == Utils.COORD_GCS:
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

        if np.linalg.norm(e_target) < self.zero_ceil:
            return 0.

        e_target = e_target / np.linalg.norm(e_target)
        wind = self.wind.value(t, x)
        wind_ortho = np.cross(e_target, wind)
        r = -wind_ortho / self.srf
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = np.arcsin(r)
        res += atan2(e_target[1], e_target[0])
        return np.array((np.cos(res), np.sin(res)))


class HTargetFB(Feedback):
    """
    Control law heading towards target
    """

    def __init__(self, target: ndarray, coords: str):
        self.target = target.copy()
        if coords == Utils.COORD_CARTESIAN:
            self.value = self._value_R2
        else:
            self.value = self._value_S2
        self.zero_ceil = 1e-3

    @classmethod
    def for_R2(cls, target: ndarray):
        cls(target, Utils.COORD_CARTESIAN)

    @classmethod
    def for_S2(cls, target: ndarray):
        cls(target, Utils.COORD_GCS)

    def _value_R2(self, t, x):
        e_target = np.zeros(2)
        e_target[:] = self.target - x

        if np.linalg.norm(e_target) < self.zero_ceil:
            return 0.

        e_target = e_target / np.linalg.norm(e_target)
        res = atan2(e_target[1], e_target[0])
        return np.array((np.cos(res), np.sin(res)))

    def _value_S2(self, t, x):
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

        if np.linalg.norm(e_target) < self.zero_ceil:
            return 0.

        e_target = e_target / np.linalg.norm(e_target)
        res = atan2(e_target[1], e_target[0])
        return np.array((np.cos(res), np.sin(res)))

    def value(self, t, x):
        pass


class MapFB(Feedback):

    def __init__(self, grid, values):
        """
        :param grid: Grid of points for discretization (nx, ny, 2)
        :param values: Heading values on the grid (nx - 1, ny - 1)
        """
        self.grid = np.array(grid)
        self.values = np.array(values)

    def value(self, _, x):
        xx = (x[0] - self.grid[0, 0, 0]) / (self.grid[-1, 0, 0] - self.grid[0, 0, 0])
        yy = (x[1] - self.grid[0, 0, 1]) / (self.grid[0, -1, 1] - self.grid[0, 0, 1])

        nx, ny, _ = self.grid.shape
        i = int((nx - 1) * xx)
        j = int((ny - 1) * yy)

        return self.values[i, j]
