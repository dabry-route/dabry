from abc import ABC
import numpy as np

from .dynamics import ZermeloDyn, PCZermeloDyn
from .wind import UniformWind, Wind
from .misc import COORD_GCS, COORD_CARTESIAN


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

    def __init__(self, v_a: float, coords=COORD_CARTESIAN):
        self.coords = coords
        super().__init__(v_a)
        self.wind = UniformWind(np.zeros(2))
        if self.coords == COORD_CARTESIAN:
            self.dyn = ZermeloDyn(self.wind, self.v_a)
        elif self.coords == COORD_GCS:
            self.dyn = PCZermeloDyn(self.wind, self.v_a)
        else:
            print(f'Unknown mode : {coords}')
            exit(1)

    def update_wind(self, wind: Wind):
        self.wind = wind
        if self.coords == COORD_CARTESIAN:
            self.dyn = ZermeloDyn(self.wind, self.v_a)
        elif self.coords == COORD_GCS:
            self.dyn = PCZermeloDyn(self.wind, self.v_a)

    def update_airspeed(self, v_a):
        super(ZermeloGeneralModel, self).update_airspeed(v_a)
        if self.coords == COORD_CARTESIAN:
            self.dyn = ZermeloDyn(self.wind, self.v_a)
        elif self.coords == COORD_GCS:
            self.dyn = PCZermeloDyn(self.wind, self.v_a)

    def __str__(self):
        return str(self.dyn) + ' with ' + str(self.wind)
