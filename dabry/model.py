from abc import ABC

import numpy as np

from dabry.misc import Utils
from .dynamics import ZermeloDyn, PCZermeloDyn
from .wind import UniformWind, Wind

"""
model.py
Navigation problem model.

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


class Model(ABC):
    """
    Defines a complete model for the UAV navigation problem.
    Defines:
        - the windfield
        - the dynamics of the model
    """

    def __init__(self, v_a: float):
        """
        :param v_a: The UAV airspeed in meters per seconds
        :param x_f: The target x-coordinate in meters
        """
        self.v_a = v_a
        self.dyn = None
        self.wind = None

    def update_airspeed(self, v_a):
        self.v_a = v_a


class ZermeloGeneralModel(Model):

    def __init__(self, v_a: float, coords=Utils.COORD_CARTESIAN):
        self.coords = coords
        super().__init__(v_a)
        self.wind = UniformWind(np.zeros(2))
        if self.coords == Utils.COORD_CARTESIAN:
            self.dyn = ZermeloDyn(self.wind, self.v_a)
        elif self.coords == Utils.COORD_GCS:
            self.dyn = PCZermeloDyn(self.wind, self.v_a)
        else:
            print(f'Unknown mode : {coords}')
            exit(1)

    def update_wind(self, wind: Wind):
        self.wind = wind
        if self.coords == Utils.COORD_CARTESIAN:
            self.dyn = ZermeloDyn(self.wind, self.v_a)
        elif self.coords == Utils.COORD_GCS:
            self.dyn = PCZermeloDyn(self.wind, self.v_a)

    def update_airspeed(self, v_a):
        super(ZermeloGeneralModel, self).update_airspeed(v_a)
        if self.coords == Utils.COORD_CARTESIAN:
            self.dyn = ZermeloDyn(self.wind, self.v_a)
        elif self.coords == Utils.COORD_GCS:
            self.dyn = PCZermeloDyn(self.wind, self.v_a)

    def __str__(self):
        return str(self.dyn) + ' with ' + str(self.wind)
