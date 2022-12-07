from abc import ABC

import numpy as np
from numpy import ndarray

import scipy.optimize


class Aero(ABC):

    def __init__(self):
        self.v_minp = None

    def power(self, airspeed):
        """
        Return required power to maintain level flight at sea level at given airspeed
        :param airspeed: Airspeed in m/s
        :return: Power in W
        """
        return 0.

    def d_power(self, airspeed):
        """
        Derivative of power function w.r.t. airspeed
        :param airspeed: Airspeed in m/s
        :return: Marginal power in W/(m/s)
        """
        # Default : finite differencing
        return (self.power(airspeed + 1e-4) - self.power(airspeed - 1e-4)) / 2e-4

    def asp_mlod(self, wind_speed):
        """
        Computes the maximum lift-over-drag ratio airspeed given longitudinal wind magnitude
        :param wind_speed: Wind speed in m/s
        :return: Airspeed in m/s
        """
        v_minp = 8  # if self.v_minp is None else self.v_minp
        f = lambda asp: self.d_power(asp) - self.power(asp) / (asp + wind_speed)
        # return scipy.optimize.brentq(
        #     lambda asp: self.d_power(asp) - self.power(asp) / (asp + wind_speed), v_minp, 100.)
        return f(scipy.optimize.brute(f, ((v_minp, 50.),))[0])

    @staticmethod
    def _vec_or_float_to_norm(x):
        if isinstance(x, ndarray):
            xn = np.linalg.norm(x)
        else:
            # Assuming float
            xn = x
        return xn

    def asp_opti(self, adjoint):
        pn = self._vec_or_float_to_norm(adjoint)
        return scipy.optimize.brentq(lambda asp: self.d_power(asp) - pn, 0.1, 100.)


class LLAero(Aero):
    """
    Lifting-line type model where polar is of the form CD = CD0 + k CL ** 2
    """

    def __init__(self):
        super().__init__()
        # Coefficients extracted from Dobrokhodov et al. 2020
        self.kp1 = 0.05
        self.kp2 = 1000
        # Coefficients from Mermoz
        # self.kp1 = 0.011
        # self.kp2 = 500
        self.v_minp = (self.kp2 / (3 * self.kp1)) ** (1/4)

    def power(self, airspeed):
        return self.kp1 * airspeed ** 3 + self.kp2 / airspeed

    def d_power(self, airspeed):
        return 3 * self.kp1 * airspeed ** 2 - self.kp2 / airspeed ** 2

    def asp_opti(self, adjoint):
        pn = self._vec_or_float_to_norm(adjoint)
        return np.sqrt(pn / (6 * self.kp1) + np.sqrt(self.kp2 / (3 * self.kp1) + pn ** 2 / (36 * self.kp1 ** 2)))
