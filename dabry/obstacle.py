import warnings
from abc import ABC, abstractmethod
from typing import Union, Optional

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

    def value(self, x):
        return self.obs.value(self.bl + x * self.scale_length)

    def d_value(self, x):
        return self.obs.d_value(self.bl + x * self.scale_length)

    def __getattr__(self, item):
        if isinstance(self.obs, DiscreteObs):
            if item == 'values':
                return self.obs.values / self.scale_length
            if item == 'bounds':
                return self.obs.bounds / self.scale_length
        return self.obs.__getattribute__(item)


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


class DiscreteObs(Obstacle):
    def __init__(self, values: ndarray, bounds: ndarray, grad_values: Optional[ndarray] = None, no_diff=False):
        super().__init__()

        if bounds.shape[0] != values.ndim:
            raise Exception(f'Incompatible shape for values and bounds: '
                            f'values has {len(values.shape)} dimensions '
                            f'so bounds must be of shape ({len(values.shape)}, 2) ({bounds.shape} given)')

        self.values = values.copy()
        self.bounds = bounds.copy()

        self.spacings = (bounds[:, 1] - bounds[:, 0]) / (np.array(self.values.shape) -
                                                         np.ones(self.values.ndim))

        self.grad_values = grad_values.copy() if grad_values is not None else None

        if self.grad_values is None and not no_diff:
            self.compute_derivatives()

    @classmethod
    def from_npz(cls, filepath, no_diff: Optional[bool] = None):
        obs = np.load(filepath, mmap_mode='r')
        kwargs = {}
        if no_diff is not None:
            kwargs['no_diff'] = no_diff
        return cls(obs['values'], obs['bounds'], **kwargs)

    @classmethod
    def from_obs(cls, obs: Obstacle, grid_bounds: Union[tuple[ndarray, ndarray], ndarray],
                 nx=100, ny=100, **kwargs):
        """
        Create discrete obstacle from analytical obstacle.
        Similar to "from_ff" of the "DiscreteFF" class
        """
        if isinstance(grid_bounds, tuple):
            if len(grid_bounds) != 2:
                raise ValueError('"grid_bounds" provided as a tuple must have two elements')
            grid_bounds = np.array(grid_bounds).transpose()

        if isinstance(grid_bounds, ndarray) and grid_bounds.shape != (2, 2):
            raise ValueError('"grid_bounds" provided as an array must have shape (2, 2)')

        bounds = grid_bounds.copy()
        shape = (nx, ny)
        spacings = (bounds[:, 1] - bounds[:, 0]) / (np.array(shape) - np.ones(bounds.shape[0]))
        values = np.zeros(shape)
        grad_values = np.zeros(shape + (2,)) if not kwargs.get('no_diff') else None
        for i in range(nx):
            for j in range(ny):
                state = bounds[-2:, 0] + np.diag((i, j)) @ spacings[-2:]
                values[i, j] = obs.value(state)
                if not kwargs.get('no_diff'):
                    grad_values[i, j, :] = obs.d_value(state)

        return cls(values, bounds, grad_values=grad_values, **kwargs)

    def value(self, x: ndarray) -> ndarray:
        return Utils.interpolate(self.values, self.bounds.transpose()[0], self.spacings, x)

    def d_value(self, x: ndarray) -> ndarray:
        return Utils.interpolate(self.grad_values, self.bounds.transpose()[0], self.spacings, x)

    def compute_derivatives(self):
        """
        Computes the derivatives of the flow field with a central difference scheme
        on the flow field native grid
        """
        grad_shape = self.values.shape + (2,)
        self.grad_values = np.zeros(grad_shape)
        inside_shape = np.array(grad_shape, dtype=int)
        inside_shape[0] -= 2  # x-axis
        inside_shape[1] -= 2  # y-axis
        inside_shape = tuple(inside_shape)

        # Use order 2 precision derivative
        self.grad_values[1:-1, 1:-1, :] = \
            np.stack(((self.values[2:, 1:-1] - self.values[:-2, 1:-1]) / (2 * self.spacings[-2]),
                      (self.values[1:-1, 2:] - self.values[1:-1, :-2]) / (2 * self.spacings[-1])),
                     axis=-1
                     ).reshape(inside_shape)
        # Padding to full grid
        # Borders
        self.grad_values[0, 1:-1, :] = self.grad_values[1, 1:-1, :]
        self.grad_values[-1, 1:-1, :] = self.grad_values[-2, 1:-1, :]
        self.grad_values[1:-1, 0, :] = self.grad_values[1:-1, 1, :]
        self.grad_values[1:-1, -1, :] = self.grad_values[1:-1, -2, :]
        # Corners
        self.grad_values[0, 0, :] = self.grad_values[1, 1, :]
        self.grad_values[0, -1, :] = self.grad_values[1, -2, :]
        self.grad_values[-1, 0, :] = self.grad_values[-2, 1, :]
        self.grad_values[-1, -1, :] = self.grad_values[-2, -2, :]


class GreatCircleObs(Obstacle):

    # TODO: validate this class
    def __init__(self, p1, p2, z1=None, z2=None, autobox=False):

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


def discretize_obs(obs: Obstacle,
                   shape: tuple[int, int],
                   bl: Optional[ndarray] = None,
                   tr: Optional[ndarray] = None):
    if not is_discrete_obstacle(obs):
        if bl is None or tr is None:
            raise Exception(f'Missing bounding box (bl, tr) to sample unbounded {obs}')
        return DiscreteObs.from_obs(obs, np.array((bl, tr)).transpose(), nx=shape[0], ny=shape[1], no_diff=True)
    else:
        if shape[0] != obs.values.shape[0] or shape[1] != obs.values.shape[1]:
            warnings.warn(f'Grid shape {shape} differs from DiscreteObs native grid. Resampling not implemented yet: '
                          'Continuing with obstacle native grid')
        return obs


def save_obs(obs: Obstacle, filepath: str,
             shape: tuple[int, int],
             bl: Optional[ndarray] = None,
             tr: Optional[ndarray] = None):
    dobs = discretize_obs(obs, shape, bl, tr)
    np.savez(filepath, values=dobs.values, bounds=dobs.bounds)


def is_discrete_obstacle(obs: Obstacle):
    return isinstance(obs, DiscreteObs) or (isinstance(obs, WrapperObs) and isinstance(obs.obs, DiscreteObs))
