from typing import Optional

import numpy as np

from .dynamics import ZermeloS2Dyn, Dynamics, ZermeloR2Dyn
from .misc import Utils, Coords
from .flowfield import UniformFF, FlowField, DiscreteFF

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


class Model:
    """
    Defines the dynamics and power law associated to a vehicle evolving in a flow field
    """

    def __init__(self, dyn: Dynamics):
        """
        :param dyn: The vehicle's dynamics
        """
        self.dyn = dyn
        self.coords = Model.which_coords(dyn.ff)

    @classmethod
    def which_coords(cls, ff: FlowField):
        return Coords.GCS if hasattr(ff, 'coords') and ff.coords == Coords.GCS else\
            Coords.CARTESIAN

    @property
    def ff(self):
        return self.dyn.ff

    @classmethod
    def zermelo_R2(cls, ff: Optional[FlowField]):
        if ff is None:
            ff = UniformFF(np.zeros(2))
        return cls(ZermeloR2Dyn(ff))

    @classmethod
    def zermelo_S2(cls, ff: FlowField):
        return cls(ZermeloS2Dyn(ff))

    @classmethod
    def zermelo(cls, ff: FlowField):
        coords = Model.which_coords(ff)
        if coords == Coords.CARTESIAN:
            return cls.zermelo_R2(ff)
        else:
            return cls.zermelo_S2(ff)
