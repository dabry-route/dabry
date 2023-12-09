from abc import ABC, abstractmethod

import numpy as np
from numpy import cos, sin
from numpy import ndarray

from dabry.misc import Utils
from dabry.wind import Wind, LinearWind

"""
dynamics.py
Models vehicle dynamics in a flow field.

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


class Dynamics(ABC):
    """
    Defines dynamics of a model.

    This class defines a function f that maps a state x, a control u, a wind vector w and a time t
    to the derivative of the state x_dot :
    x_dot = f(x, u, w, t)
    """

    def __init__(self,
                 wind: Wind):
        self.wind = wind

    @abstractmethod
    def value(self,
              x: ndarray,
              u: ndarray,
              t: float,
              v_a=None) -> ndarray:
        """
        Computes the derivative of the state relative to time in the given model

        :param x: The state vector
        :param u: The control vector
        :param t: The time
        :return: dx/dt
        """
        pass

    @abstractmethod
    def d_value__d_state(self,
                         x: ndarray,
                         u: ndarray,
                         t: float,
                         v_a=None) -> ndarray:
        """
        Computes the derivative of f w.r.t. the state x
        :param x: The state vector
        :param u: The control vector
        :param t: The time
        :return: df/dx (partial derivative)
        """
        return self.wind.d_value(x)


class ZermeloDyn(Dynamics):
    """
    Zermelo dynamics.

    The state vector is the 2D vector (x,y) in R^2.
    The control is the real-valued heading of the drone theta in R
    """

    def __init__(self, wind: Wind, v_a: float):
        """
        :param v_a: Aircraft speed relative to the air in m/s
        """
        super().__init__(wind)
        self.v_a = v_a

    def value(self, x, u, t, v_a=None):
        if v_a is None:
            v_a = self.v_a
        return v_a * np.array([np.cos(u), np.sin(u)]) + self.wind.value(t, x)

    def d_value__d_state(self, x, u, t, v_a=None):
        return self.wind.d_value(t, x)

    def __str__(self):
        return 'Zermelo dynamics'


class PCZermeloDyn(Dynamics):
    """
    Zermelo dynamics for plate-carree spherical projection.

    The state vector is the 2D vector (x,y) in R^2.
    x is the longitude and y the latitude.
    The control is the real-valued heading of the drone.
    """

    def __init__(self, wind: Wind, v_a: float):
        """
        :param v_a: Aircraft speed relative to the air in m/s
        """
        super().__init__(wind)
        self.v_a = v_a
        self.factor = 1 / Utils.EARTH_RADIUS

    def value(self, x, psi, t, v_a=None):
        if v_a is None:
            v_a = self.v_a
        return self.factor * (
                np.diag([1 / cos(x[1]), 1.]) @ (v_a * np.array([sin(psi), cos(psi)]) + self.wind.value(t, x)))

    def d_value__d_state(self, x, psi, t, v_a=None):
        if v_a is None:
            v_a = self.v_a
        wind_gradient = np.zeros((x.size, x.size))
        wind_gradient[:] = self.wind.d_value(t, x)
        res = self.factor * np.column_stack((np.diag([1 / cos(x[1]), 1.]) @ wind_gradient[:, 0],
                                             np.diag([sin(x[1]) / (cos(x[1]) ** 2), 0.]) @
                                             (v_a * np.array([sin(psi), cos(psi)]) + self.wind.value(t, x)) +
                                             np.diag([1 / cos(x[1]), 1.]) @ wind_gradient[:, 1]))
        return res

    def __str__(self):
        return 'Zermelo dynamics on plate carree projection'


if __name__ == '__main__':
    wind = LinearWind(np.array([[1., 2.],
                                [-3., 4.]]), np.array([0., 0.]), np.array([0., 0.]))
    pczd = PCZermeloDyn(wind, 1.2)
    print(pczd.value(np.array([2., 3.]), 0., 0.))
    print(pczd.value(np.array([0., 0.]), 0., 0.))
    print(pczd.value(np.array([0., 0.]), 0.5, 0.))
    print(pczd.value(np.array([0., 0.]), np.pi / 2., 0.))
    print(pczd.d_value__d_state(np.array([0., np.pi / 2. - 1e-3]), np.pi / 2., 0.))