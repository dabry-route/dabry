from abc import abstractmethod

import h5py
import numpy as np
from numpy import ndarray
from scipy.interpolate import RegularGridInterpolator

"""
penalty.py
Penalty object to handle penalty fields

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


class Penalty:

    def __init__(self, length_scale=None):
        self._dx = 1e-8 * length_scale if length_scale is not None else 1e-8

    @abstractmethod
    def value(self, t, x):
        pass

    def d_value(self, t, x):
        # Finite differencing by default
        dx = self._dx
        a1 = 1 / (2 * dx) * (self.value(t, x + dx * np.array((1, 0))) - self.value(t, x - dx * np.array((1, 0))))
        a2 = 1 / (2 * dx) * (self.value(t, x + dx * np.array((0, 1))) - self.value(t, x - dx * np.array((0, 1))))
        return np.hstack((a1, a2))


class WrapperPen(Penalty):
    """
    Wrap a penalty to work with appropriate units
    """

    def __init__(self, penalty: Penalty, scale_length: float, bl: ndarray,
                 scale_time: float, time_origin: float):
        super().__init__()
        self.penalty: Penalty = penalty
        self.scale_length = scale_length
        self.bl: ndarray = bl.copy()
        self.scale_time = scale_time
        self.time_origin = time_origin
        self.scale_speed = scale_length / scale_time
        self._scaler_speed = 1 / self.scale_speed

    def value(self, t, x):
        return self._scaler_speed * self.penalty.value(
            self.time_origin + t * self.scale_time, self.bl + x * self.scale_length)

    def d_value(self, t, x):
        return self._scaler_speed * self.penalty.d_value(
            self.time_origin + t * self.scale_time, self.bl + x * self.scale_length)


class NullPenalty(Penalty):

    def __init__(self):
        super(NullPenalty, self).__init__()

    def value(self, t, x):
        return 0.

    def d_value(self, t, x):
        return np.array((0., 0.))


class CirclePenalty(Penalty):

    def __init__(self, center: ndarray, radius: float, amplitude: float):
        super().__init__()
        self.center = center.copy()
        self.radius = radius
        self.amplitude = amplitude

    def value(self, t, x):
        return np.max((self.amplitude * 0.5 * (self.radius ** 2 - np.sum(np.square(x - self.center))), 0.))


class DiscretePenalty(Penalty):

    # TODO: validate this class
    def __init__(self, *args):
        super().__init__()
        self.data = None
        self.ts = None
        self.grid = None
        self.itp = None
        if len(args) != 0:
            if len(args) != 3:
                raise Exception('Requires 3 arguments : data, timestamps, grid')
            data, ts, grid = args[0], args[1], args[2]
            self.data = np.array(data)
            self.ts = np.array(ts)
            self.grid = np.array(grid)
            self._setup()

    def value(self, t, x):
        u = np.zeros(3)
        u[0] = t
        u[1:] = x
        return self.itp(u)

    def _setup(self):
        nx, ny, _ = self.grid.shape
        x = np.zeros(nx)
        y = np.zeros(ny)
        x[:] = self.grid[:, 0, 0]
        y[:] = self.grid[0, :, 1]
        self.itp = RegularGridInterpolator((self.ts, x, y), self.data, method='linear', bounds_error=False,
                                           fill_value=0.)

    def load(self, filepath):
        with h5py.File(filepath, 'r') as f:
            self.data = np.array(f['data'])
            self.ts = np.array(f['ts'])
            self.grid = np.array(f['grid'])
        self._setup()
