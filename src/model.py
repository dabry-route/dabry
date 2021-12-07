from abc import ABC

import numpy as np
from numpy import ndarray

from src.dynamics import ZermeloDyn
from src.wind import TSEqualWind, VortexWind, SourceWind, VortexUniformWind, UniformWind, Wind


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


class Model1(Model):
    """
    Zermelo problem with two sectors of constant, y-axis oriented wind
    """

    def __init__(self,
                 v_a: float,
                 v_w1: float,
                 v_w2: float,
                 x_f: float):
        """
        :param v_a: UAV airspeed in m/s
        :param v_w1: Wind value in first sector
        :param v_w2: Wind value in second sector
        :param x_f: Target x-coordinate
        """
        super().__init__(v_a, x_f)
        self.v_w1 = v_w1
        self.v_w2 = v_w2
        self.wind = TSEqualWind(v_w1, v_w2, x_f)
        self.dyn = ZermeloDyn(self.wind, self.v_a)


class Model2(Model):
    """
    Zermelo problem with vortex wind
    """

    def __init__(self,
                 v_a: float,
                 x_f: float,
                 omega: ndarray,
                 gamma: float):
        super().__init__(v_a, x_f)
        self.omega = omega
        self.gamma = gamma
        self.wind = VortexWind(omega[0], omega[1], gamma)
        self.dyn = ZermeloDyn(self.wind, self.v_a)


class Model3(Model):
    """
    Zermelo problem with source wind
    """

    def __init__(self,
                 v_a: float,
                 x_f: float,
                 omega: ndarray,
                 flux: float):
        super().__init__(v_a, x_f)
        self.omega = omega
        self.flux = flux
        self.wind = SourceWind(omega[0], omega[1], flux)
        self.dyn = ZermeloDyn(self.wind, self.v_a)


class Model4(Model):
    def __init__(self,
                 v_a: float,
                 x_f: float,
                 const_wind: ndarray,
                 omega: ndarray,
                 gamma: float):
        super().__init__(v_a, x_f)
        self.omega = omega
        self.gamma = gamma
        self.wind = VortexUniformWind(const_wind, omega[0], omega[1], gamma)
        self.dyn = ZermeloDyn(self.wind, self.v_a)


class Model5(Model):
    def __init__(self,
                 v_a: float,
                 x_f: float,
                 const_wind: ndarray,
                 omega: ndarray,
                 gamma: float):
        super().__init__(v_a, x_f)
        self.omega = omega
        self.gamma = gamma
        self.wind = UniformWind(const_wind) + VortexWind(omega[0], omega[1], gamma)
        self.dyn = ZermeloDyn(self.wind, self.v_a)


class ZermeloGeneralModel(Model):

    def __init__(self,
                 v_a: float,
                 x_f: float):
        super().__init__(v_a, x_f)
        self.wind = UniformWind(np.zeros(2))
        self.dyn = ZermeloDyn(self.wind, self.v_a)

    def update_wind(self, wind: Wind):
        self.wind = wind
        self.dyn = ZermeloDyn(self.wind, self.v_a)
