import numpy as np

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

    def __init__(self, value_func, d_value_func=None, length_scale=None):
        self._dx = 1e-10 * length_scale if length_scale is not None else 1e-10
        self.value = value_func
        if d_value_func is not None:
            self.d_value = d_value_func

    def value(self, x):
        pass

    def d_value(self, x):
        # Finite differencing by default
        dx = self._dx
        a1 = 1 / (2 * dx) * (self.value(x + dx * np.array((1, 0))) - self.value(x - dx * np.array((1, 0))))
        a2 = 1 / (2 * dx) * (self.value(x + dx * np.array((0, 1))) - self.value(x - dx * np.array((0, 1))))
        return np.array((a1, a2))
