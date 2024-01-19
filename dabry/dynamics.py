from abc import ABC, abstractmethod

import numpy as np
from numpy import cos, sin
from numpy import ndarray

from dabry.misc import Utils
from dabry.flowfield import FlowField

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

    This class defines a function f that maps a time t, a state x and a control u
    to the derivative of the state x_dot :
    x_dot = f(t, x, u)
    """

    def __init__(self, ff: FlowField):
        self.ff = ff

    @abstractmethod
    def value(self, t: float, x: ndarray, u: ndarray) -> ndarray:
        """
        Computes the derivative of the state relative to time in the given model
        :param x: The state vector
        :param u: The control vector
        :param t: The time
        :return: dx/dt
        """
        pass

    @abstractmethod
    def d_value__d_state(self, t: float, x: ndarray, u: ndarray) -> ndarray:
        """
        Computes the derivative of f w.r.t. the state x
        :param x: The state vector
        :param u: The control vector
        :param t: The time
        :return: df/dx (partial derivative)
        """
        pass

    def __call__(self, t: float, x: ndarray, u: ndarray) -> ndarray:
        return self.value(t, x, u)


class ZermeloR2Dyn(Dynamics):
    """
    Zermelo dynamics in planar R^2 space
    """

    def __init__(self, ff: FlowField):
        super().__init__(ff)

    def value(self, t, x, u):
        return u + self.ff.value(t, x)

    def d_value__d_state(self, t, x, u):
        return self.ff.d_value(t, x)


class ZermeloS2Dyn(Dynamics):
    """
    Zermelo dynamics over plate-carrée projection of the sphere S^2.
    Coordinates are (longitude, latitude).
    """

    def __init__(self, ff: FlowField):
        super().__init__(ff)

    def value(self, t, x, u):
        return np.diag([1 / cos(x[1]), 1.]) @ (u + self.ff.value(t, x))

    def d_value__d_state(self, t, x, u):
        return np.diag((1 / cos(x[1]), 1)) @ self.ff.d_value(t, x) + \
               np.column_stack((np.zeros(2), np.diag((sin(x[1]) / (cos(x[1]) ** 2), 0.)) @ u + self.ff.value(t, x)))

