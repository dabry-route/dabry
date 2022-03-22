from abc import ABC

import numpy as np

from src.dynamics import ZermeloDyn, PCZermeloDyn
from src.wind import UniformWind, Wind
from src.misc import COORD_GCS, COORD_CARTESIAN


class Model(ABC):
    """
    Defines a complete model for the UAV navigation problem.
    Defines:
        - the windfield
        - the dynamics of the model
    """

    def __init__(self, v_a: float, x_f: float):
        """
        :param v_a: The UAV airspeed in meters per seconds
        :param x_f: The target x-coordinate in meters
        """
        self.v_a = v_a
        self.x_f = x_f
        self.dyn = None
        self.wind = None


class ZermeloGeneralModel(Model):

    def __init__(self,
                 v_a: float,
                 x_f: float,
                 mode=COORD_CARTESIAN):
        self.mode = mode
        super().__init__(v_a, x_f)
        self.wind = UniformWind(np.zeros(2))
        if self.mode == COORD_CARTESIAN:
            self.dyn = ZermeloDyn(self.wind, self.v_a)
        elif self.mode == COORD_GCS:
            self.dyn = PCZermeloDyn(self.wind, self.v_a)
        else:
            print(f'Unknown mode : {mode}')
            exit(1)

    def update_wind(self, wind: Wind):
        self.wind = wind
        if self.mode == COORD_CARTESIAN:
            self.dyn = ZermeloDyn(self.wind, self.v_a)
        elif self.mode == COORD_GCS:
            self.dyn = PCZermeloDyn(self.wind, self.v_a)

    def __str__(self):
        return str(self.dyn) + ' with ' + str(self.wind)
