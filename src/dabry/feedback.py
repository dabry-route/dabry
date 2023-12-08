import random
from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize
from numpy import arctan2 as atan2
from numpy import ndarray
from numpy import pi, sin, cos

from dabry.aero import Aero
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

    def __init__(self,
                 dimension: int,
                 wind: Wind):
        """
        :param dimension: The dimension of the control
        :param wind: The wind conditions
        """
        self.dim = dimension
        self.wind = wind

    @abstractmethod
    def value(self, t: float, x: ndarray) -> ndarray:
        """
        :param t: The time at which to compute the control
        :param x: The state
        :return: The feedback
        """
        pass


class AirspeedLaw(ABC):
    """
    Defines an airspeed law
    """

    @abstractmethod
    def value(self, t: float, x: ndarray) -> float:
        """
        :param t: The time at which to compute the airspeed
        :param x: The state
        :return: The corresponding airspeed
        """
        pass


class MultiFeedback(ABC):
    """
    Defines simultaneously heading and airspeed laws
    """

    @abstractmethod
    def value(self, t: float, x: ndarray) -> float:
        """
        :param t: The time at which to compute the feedback
        :param x: The state
        :return: Heading (rad), Airspeed (m/s)
        """
        pass


class ZermeloPMPFB(Feedback):

    def __init__(self,
                 wind: Wind,
                 v_a: float,
                 p_y: float):
        """
        :param v_a: UAV airspeed in m/s
        :param p_y: Feedback parameter (from PMP)
        """
        super().__init__(1, wind)
        self.v_a = v_a
        self.p_y = p_y

    def value(self, t, x):
        wv = self.wind.value(x)[1]
        sin_value = -self.p_y * self.v_a / (1 + self.p_y * wv)
        if sin_value > 1.:
            raise ValueError("ZermeloPMPFB feedback value : sin value above 1")
        elif sin_value < -1.:
            raise ValueError("ZermeloPMPFB feedback value : sin value below -1")
        return np.arcsin(sin_value)


class ConstantFB(Feedback):

    def __init__(self, value):
        super().__init__(1, Wind())
        self._value = value

    def value(self, t, x):
        return self._value


class RandomFB(Feedback):

    def __init__(self, lower, upper, seed=42):
        self.lower = lower
        self.upper = upper
        random.seed(seed)
        super().__init__(1, Wind())

    def value(self, t, x):
        return random.uniform(self.lower, self.upper)


class FixedHeadingFB(Feedback):
    """
    Defines a control law to follow a straight line from initial position
    defined by its angle to the x-axis.
    When the wind value is too high for the UAV to keep its line, the
    control law steers perpendicular to the target direction
    """

    def __init__(self, wind, v_a: float, initial_steering: float, coords: str):
        """
        :param v_a: The UAV airspeed in m/s
        :param initial_steering: The heading to keep constant in rad
        """
        super().__init__(1, wind)
        self.v_a = v_a
        self.theta_0 = initial_steering
        self.coords = coords

    def value(self, t, x):
        if self.coords == Utils.COORD_CARTESIAN:
            e_theta_0 = np.array([np.cos(self.theta_0), np.sin(self.theta_0)])
        else:
            # coords gcs
            e_theta_0 = np.array([np.sin(self.theta_0), np.cos(self.theta_0)])
        wind = self.wind.value(t, x)
        wind_ortho = np.cross(e_theta_0, wind)
        r = -wind_ortho / self.v_a
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = np.arcsin(r)
            if self.coords == Utils.COORD_GCS:
                res *= -1
        res += self.theta_0
        return res


class GreatCircleFB(Feedback):
    """
    Control law for GCS problems only.
    Tries to stay on a great circle when wind allows it.
    """

    def __init__(self, wind, v_a, target):
        super().__init__(1, wind)
        self.v_a = v_a
        self.lon_t, self.lat_t = target

    def value(self, t, x: ndarray) -> ndarray:
        # First get the desired heading to follow great circle
        lon, lat = x
        lon_t, lat_t = self.lon_t, self.lat_t
        u0 = np.arctan(1. / (np.cos(lat) * np.tan(lat_t) / np.sin(lon_t - lon) - np.sin(lat) / np.tan(lon_t - lon)))

        e_0 = np.array([np.sin(u0), np.cos(u0)])
        wind = self.wind.value(t, x)
        wind_ortho = np.cross(e_0, wind)
        r = -wind_ortho / self.v_a
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = -np.arcsin(r)
        res += u0
        return res


class GSTargetFB(Feedback):
    """
    Control law trying to put ground speed vector towards a fixed target
    """

    def __init__(self, wind, v_a: float, target: ndarray, coords: str):
        super().__init__(1, wind)
        self.v_a = v_a
        self.target = np.zeros(2)
        self.target[:] = target
        self.coords = coords
        self.zero_ceil = 1e-3

    def value(self, t, x: ndarray):
        if self.coords == Utils.COORD_GCS:
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
        r = -wind_ortho / self.v_a
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = np.arcsin(r)
        res += atan2(e_target[1], e_target[0])
        if self.coords == Utils.COORD_GCS:
            res = pi / 2 - res
        return res


class HTargetFB(Feedback):
    """
    Control law heading towards target
    """

    def __init__(self, target: ndarray, coords: str):
        super().__init__(1, UniformWind(np.array((0., 0.))))
        self.target = np.zeros(2)
        self.target[:] = target
        self.coords = coords
        self.zero_ceil = 1e-3

    def value(self, t, x: ndarray):
        # Assuming GCS
        if self.coords == Utils.COORD_GCS:
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
        else:
            #  self.coords == COORD_CARTESIAN
            e_target = np.zeros(2)
            e_target[:] = self.target - x

        if np.linalg.norm(e_target) < self.zero_ceil:
            return 0.

        e_target = e_target / np.linalg.norm(e_target)
        res = atan2(e_target[1], e_target[0])
        if self.coords == Utils.COORD_GCS:
            res = pi / 2 - res
        return res


class WindAlignedFB(Feedback):
    """
    Defines a control law to align the airspeed vector with the wind vector
    """

    def __init__(self, wind):
        super().__init__(1, wind)

    def value(self, t, x):
        wind = self.wind.value(x)
        theta = np.arctan2(wind[1], wind[0])
        return theta


class FunFB(Feedback):

    def __init__(self, value_func, no_time=False):
        super().__init__(1, Wind())
        self._value_func = value_func
        self.no_time = no_time

    def value(self, t, x):
        if self.no_time:
            return self._value_func(x)
        else:
            return self._value_func(t, x)


class ConstantAS(AirspeedLaw):

    def __init__(self, airspeed):
        self.airspeed = airspeed

    def value(self, t, _):
        return self.airspeed


class ParamAS(AirspeedLaw):

    def __init__(self, aspd, t_end, t_start=0., interp='linear'):
        """
        Airspeed law interpreted as sampled functional data points
        at evenly spaced timestamps between t_start and t_end
        :param aspd: List of airspeed values in meters per second
        :param t_end: Duration of time window
        :param t_start: Start date if needed
        :param interp: Interpolation mode (only 'linear')
        """
        self.airspeed = np.zeros(aspd.shape)
        self.airspeed[:] = aspd
        self.t_end = t_end
        self.t_start = t_start
        self.interp = interp

    def _index(self, t):
        nt = self.airspeed.shape[0]
        if t <= self.t_start:
            return 0, 0.
        elif t >= self.t_end:
            return nt - 2, 1.
        tau = (t - self.t_start) / (self.t_end - self.t_start)
        k = int(tau * (nt - 1))
        return k, tau * (nt - 1) - k

    def value(self, t, _):
        k, alpha = self._index(t)
        return (1 - alpha) * self.airspeed[k] + alpha * self.airspeed[k + 1]


class FunAS(AirspeedLaw):

    def __init__(self, value_func):
        self.value_func = value_func

    def value(self, t, x):
        return self.value_func(t, x)


class E_GSTargetFB(MultiFeedback):

    def __init__(self, wind: Wind, aero: Aero, target: ndarray, coords: str):
        self.wind = wind
        self.aero = aero
        self.target = target
        self.coords = coords
        self.zero_ceil = 1e-3
        self.wind_ortho = None
        self.angle0 = None

    def update(self, t, x):
        if self.coords == Utils.COORD_GCS:
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
        self.wind_ortho = np.cross(e_target, wind)
        self.angle0 = atan2(e_target[1], e_target[0])

    def _heading(self, v_a):
        r = -self.wind_ortho / v_a
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = np.arcsin(r)
        res += self.angle0
        v_heading = np.array((np.cos(res), np.sin(res)))
        if self.coords == Utils.COORD_GCS:
            res = pi / 2 - res
        return res, v_heading

    def value(self, t, x):
        self.update(t, x)
        asp_opti = scipy.optimize.brentq(
            lambda asp: self.aero.d_power(asp) * (
                    asp + self._heading(asp)[1] @ self.wind.value(t, x)) - self.aero.power(
                asp), max(self.wind_ortho, self.aero.v_minp), 100.)
        return self._heading(asp_opti)[0], asp_opti


class GriddedStaticFB(Feedback):

    def __init__(self, grid, values):
        """
        :param grid: Grid of points for discretization (nx, ny, 2)
        :param values: Heading values and validity date on the grid (nx - 1, ny - 1, 2)
        """
        self.grid = np.array(grid)
        self.values = np.array(values)
        super().__init__(2, UniformWind(np.array((0, 0))))

    def value(self, _, x):
        xx = (x[0] - self.grid[0, 0, 0]) / (self.grid[-1, 0, 0] - self.grid[0, 0, 0])
        yy = (x[1] - self.grid[0, 0, 1]) / (self.grid[0, -1, 1] - self.grid[0, 0, 1])

        nx, ny, _ = self.grid.shape
        i = int((nx - 1) * xx)
        j = int((ny - 1) * yy)

        return self.values[i, j]


class GriddedFB(Feedback):

    def __init__(self, grid, values):
        """
        :param grid: Grid of points for discretization (nx, ny, 2)
        :param values: Heading values and validity date on the grid, shape (nx - 1, ny - 1, ndt, 2)
        grid cell index, control value layer, (value of control, validity date)
        """
        self.grid = np.array(grid)
        self.values = np.array(values)
        super().__init__(2, UniformWind(np.array((0, 0))))

    def value(self, t, x):
        xx = (x[0] - self.grid[0, 0, 0]) / (self.grid[-1, 0, 0] - self.grid[0, 0, 0])
        yy = (x[1] - self.grid[0, 0, 1]) / (self.grid[0, -1, 1] - self.grid[0, 0, 1])

        nx, ny, _ = self.grid.shape
        i = int((nx - 1) * xx)
        j = int((ny - 1) * yy)

        ndt = self.values.shape[2]
        if t < self.values[i, j, 0, 1] or self.values[i, j, ndt - 1, 1] < t:
            msg = 'Control law : ts {t:.0f} out of scope ({tl:.0f}, {tu:.0f})'.format(t=t, tl=self.values[i, j, 0, 1],
                                                                                      tu=self.values[i, j, ndt - 1, 1])
            raise Exception(msg)
        k_inf = 0
        for k in range(ndt - 1):
            k_inf = k
            if self.values[i, j, k, 1] < t:
                break
        alpha = (t - self.values[i, j, k_inf, 1]) / (self.values[i, j, k_inf + 1, 1] - self.values[i, j, k_inf, 1])
        return (1 - alpha) * self.values[i, j, k_inf, 0] + alpha * self.values[i, j, k_inf + 1, 0]
