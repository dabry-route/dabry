from abc import ABC, abstractmethod

import numpy as np
from wind import Wind
from numpy import ndarray


class Dynamics(ABC):
    """
    Defines dynamics of a model.

    This class defines a function f that maps a state x, a control u, a wind vector w and a time t
    to the derivative of the state x_dot :
    x_dot = f(x, u, w, t)
    """

    def __init__(self,
                 wind: Wind):
        self.wind = wind

    @abstractmethod
    def value(self,
              x: ndarray,
              u: ndarray,
              t: float) -> ndarray:
        """
        Computes the derivative of the state relative to time in the given model

        :param x: The state vector
        :param u: The control vector
        :param t: The time
        :return: dx/dt
        """
        pass

    @abstractmethod
    def d_value__d_state(self,
                         x: ndarray,
                         u: ndarray,
                         t: float) -> ndarray:
        """
        Computes the derivative of f w.r.t. the state x
        :param x: The state vector
        :param u: The control vector
        :param t: The time
        :return: df/dx (partial derivative)
        """
        return self.wind.d_value(x)


class ZermeloDyn(Dynamics):
    """
    Zermelo dynamics.

    The state vector is the 2D vector (x,y) in R^2.
    The control is the real-valued heading of the drone theta in R
    """

    def __init__(self, wind: Wind, v_a: float):
        """
        :param v_a: Aircraft speed relative to the air in m/s
        """
        super().__init__(wind)
        self.v_a = v_a

    def value(self, x, u, t):
        return self.v_a * np.array([np.cos(u), np.sin(u)]) + self.wind.value(x)

    def d_value__d_state(self, x, u, t):
        return self.wind.d_value(x)

    def __str__(self):
        return 'Zermelo dynamics'
