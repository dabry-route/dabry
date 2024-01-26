from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy import ndarray

from dabry.misc import Utils, terminal

"""
obstacle.py
Obstacle definition as real-valued function of space for both
planar and spherical cases.

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


class Obstacle(ABC):

    def __init__(self):
        pass

    @terminal
    def event(self, time: float, state_aug: ndarray):
        return self.value(state_aug[:2])

    @abstractmethod
    def value(self, x: ndarray):
        """
        Return a negative value if within obstacle, positive if outside and zero at the border
        :param x: Position at which to get value (1D numpy array)
        :return: Obstacle function value
        """
        pass

    def d_value(self, x: ndarray) -> ndarray:
        """
        Derivative of obstacle value function
        :param x: Position at which to get derivative (1D numpy array)
        :return: Gradient of obstacle function at point
        """
        # Finite differencing, centered scheme
        eps = 1e-8
        n = x.shape[0]
        emat = np.diag(n * (eps,))
        grad = np.zeros(n)
        for i in range(n):
            grad[i] = (self.value(x + emat[i]) - self.value(x - emat[i])) / (2 * eps)
        return grad


class WrapperObs(Obstacle):
    """
    Wrap an obstacle to work with appropriate units
    """

    def __init__(self, obs: Obstacle, scale_length: float, bl: ndarray):
        super().__init__()
        self.obs: Obstacle = obs
        self.scale_length = scale_length
        self.bl: ndarray = bl.copy()
        # Multiplication will be quicker than division
        self._scaler_length = 1 / self.scale_length

    def value(self, x):
        return self.obs.value((x - self.bl) * self._scaler_length)

    def d_value(self, x):
        return self.obs.d_value((x - self.bl) * self._scaler_length)


class CircleObs(Obstacle):
    """
    Circle obstacle defined by center and radius
    """

    def __init__(self, center: Union[ndarray, tuple[float, float]], radius: float):
        self.center = center.copy() if isinstance(center, ndarray) else np.array(center)
        self.radius = radius
        self._sqradius = radius ** 2
        super().__init__()

    def value(self, x: ndarray):
        return 0.5 * (np.sum(np.square(x - self.center)) - self._sqradius)

    def d_value(self, x: ndarray) -> ndarray:
        return x - self.center


class EightFrameObs(Obstacle):
    """
    Smooth rectangle frame based on the eightth norm ||x||_8
    """
    def __init__(self, bl: ndarray, tr: ndarray):
        self.bl = bl.copy()
        self.tr = tr.copy()
        self.center = 0.5 * (self.bl + self.tr)
        self.scaler = np.diag(1 / (0.5 * (self.tr - self.bl)))
        super().__init__()

    def value(self, x: ndarray):
        return (1 / 8) * (1. - np.sum(np.power(np.dot(x[..., :2] - self.center, self.scaler), 8), -1))

    def d_value(self, x: ndarray) -> ndarray:
        return np.power(np.dot(x[..., :2] - self.center, self.scaler), 7)


class FrameObs(Obstacle):
    """
    Rectangle obstacle acting as a frame
    """

    def __init__(self, bl: ndarray, tr: ndarray):
        self.bl = bl.copy()
        self.tr = tr.copy()
        self.center = 0.5 * (bl + tr)
        self.scaler = np.diag(1 / (0.5 * (self.tr - self.bl)))
        super().__init__()

    def value(self, x):
        return 1. - np.max(np.abs(np.dot(x[..., :2] - self.center, self.scaler)), -1)

    def d_value(self, x):
        xx = np.dot(x[..., :2] - self.center, self.scaler)
        c1 = np.dot(xx, np.array((1., 1.)))
        c2 = np.dot(xx, np.array((1., -1.)))
        g1 = np.dot(np.ones(xx.shape), np.diag((1., 0.)))
        g2 = np.dot(np.ones(xx.shape), np.diag((0., 1.)))
        g3 = np.dot(np.ones(xx.shape), np.diag((-1., 0.)))
        g4 = np.dot(np.ones(xx.shape), np.diag((0., -1.)))
        return np.where((c1 > 0) * (c2 > 0), g1,
                        np.where((c1 > 0) * (c2 < 0), g2,
                                 np.where((c1 < 0) * (c2 < 0), g3,
                                          g4)))


class GreatCircleObs(Obstacle):

    def __init__(self, p1, p2, z1=None, z2=None, autobox=False):
        # TODO: validate this class
        # Cross product of p1 and p2 points TOWARDS obstacle
        # z1 and z2 are zone limiters
        X1 = np.array((np.cos(p1[0]) * np.cos(p1[1]),
                       np.sin(p1[0]) * np.cos(p1[1]),
                       np.sin(p1[1])))
        X2 = np.array((np.cos(p2[0]) * np.cos(p2[1]),
                       np.sin(p2[0]) * np.cos(p2[1]),
                       np.sin(p2[1])))
        if not autobox:
            self.z1 = z1
            self.z2 = z2
        else:
            delta_lon = Utils.angular_diff(p1[0], p2[0])
            delta_lat = p1[1] - p2[0]
            self.z1 = np.array((min(p1[0] - delta_lon / 2., p2[0] - delta_lon / 2.),
                                min(p1[1] - delta_lat / 2., p2[1] - delta_lat / 2.)))
            self.z2 = np.array((max(p1[0] + delta_lon / 2., p2[0] + delta_lon / 2.),
                                max(p1[1] + delta_lat / 2., p2[1] + delta_lat / 2.)))

        self.dir_vect = -np.cross(X1, X2)
        self.dir_vect /= np.linalg.norm(self.dir_vect)
        super().__init__()

    def value(self, x):
        if self.z1 is not None:
            if not Utils.in_lonlat_box(self.z1, self.z2, x):
                return 1.
        X = np.array((np.cos(x[0]) * np.cos(x[1]), np.sin(x[0]) * np.cos(x[1]), np.sin(x[1])))
        return X @ self.dir_vect

    def d_value(self, x):
        if self.z1 is not None:
            if not Utils.in_lonlat_box(self.z1, self.z2, x):
                return np.array((1., 1.))
        d_dphi = np.array((-np.sin(x[0]) * np.cos(x[1]), np.cos(x[0]) * np.cos(x[1]), 0))
        d_dlam = np.array((-np.cos(x[0]) * np.sin(x[1]), -np.sin(x[0]) * np.sin(x[1]), np.cos(x[1])))
        return np.array((self.dir_vect @ d_dphi, self.dir_vect @ d_dlam))
