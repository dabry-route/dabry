from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
import random

from src.wind import Wind


class Feedback(ABC):
    """
    Defines feedback laws for given models
    """

    def __init__(self,
                 dimension: int,
                 wind: Wind):
        """
        :param dimension: The dimension of the control
        :param wind: The wind conditions
        """
        self.dim = dimension
        self.wind = wind

    @abstractmethod
    def value(self, x: ndarray) -> ndarray:
        """
        :param x: The state
        :return: The feedback
        """
        pass


class ZermeloPMPFB(Feedback):

    def __init__(self,
                 wind: Wind,
                 v_a: float,
                 p_y: float):
        """
        :param v_a: UAV airspeed in m/s
        :param p_y: Feedback parameter (from PMP)
        """
        super().__init__(1, wind)
        self.v_a = v_a
        self.p_y = p_y

    def value(self, x):
        wv = self.wind.value(x)[1]
        sin_value = -self.p_y * self.v_a / (1 + self.p_y * wv)
        if sin_value > 1.:
            raise ValueError("ZermeloPMPFB feedback value : sin value above 1")
        elif sin_value < -1.:
            raise ValueError("ZermeloPMPFB feedback value : sin value below -1")
        return np.arcsin(sin_value)


class ConstantFB(Feedback):

    def __init__(self, value):
        super().__init__(1, None)
        self._value = value

    def value(self, x):
        return self._value


class RandomFB(Feedback):

    def __init__(self, lower, upper, seed=42):
        self.lower = lower
        self.upper = upper
        random.seed(seed)
        super().__init__(1, None)

    def value(self, x):
        return random.uniform(self.lower, self.upper)


class FixedHeadingFB(Feedback):
    """
    Defines a control law to follow a straight line from initial position
    defined by its angle to the x-axis.
    Behavior uncertain when entering strong wind zones (wind value
    greater than airpseed)
    """

    def __init__(self, wind, v_a: float, initial_steering: float):
        """
        :param v_a: The UAV airspeed in m/s
        :param initial_steering: The heading to keep constant in rad
        """
        super().__init__(1, wind)
        self.v_a = v_a
        self.theta_0 = initial_steering

    def value(self, x):
        e_theta_0 = np.array([np.cos(self.theta_0), np.sin(self.theta_0)])
        wind = self.wind.value(x)
        wind_tangent = np.cross(e_theta_0, wind)
        r = -wind_tangent / self.v_a
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = np.arcsin(-wind_tangent / self.v_a)
        res += self.theta_0
        return res
