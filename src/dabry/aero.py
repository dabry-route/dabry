from abc import ABC

import numpy as np
import scipy.optimize
from numpy import ndarray

"""
aero.py
Models aerodynamics for flight power consumption.

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


class Aero(ABC):

    def __init__(self):
        self.v_minp = None

    def power(self, _):
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

    def __init__(self, mode='dobro'):
        super().__init__()
        self.mode = mode
        if mode == 'dobro':
            # Coefficients extracted from Dobrokhodov et al. 2020
            self.kp1 = 0.05
            self.kp2 = 1000
        elif mode == 'dabry':
            # Coefficients from Mermoz
            self.kp1 = 0.011
            self.kp2 = 500
        else:
            raise Exception(f'Unknown mode {mode}')
        self.v_minp = (self.kp2 / (3 * self.kp1)) ** (1 / 4)

    def power(self, airspeed):
        return self.kp1 * airspeed ** 3 + self.kp2 / airspeed

    def d_power(self, airspeed):
        return 3 * self.kp1 * airspeed ** 2 - self.kp2 / airspeed ** 2

    def asp_opti(self, adjoint):
        pn = self._vec_or_float_to_norm(adjoint)
        return np.sqrt(pn / (6 * self.kp1) + np.sqrt(self.kp2 / (3 * self.kp1) + pn ** 2 / (36 * self.kp1 ** 2)))


class MermozAero(Aero):
    """
    Model P_req = A0 V ^ 3 + A1 V + A2 1 / V from fitting CD = a0 CL ^ 2 + a1 CL + a2 to real polar
    """

    def __init__(self):
        super().__init__()
        self.A0 = 0.0152
        self.A1 = -6.1139
        self.A2 = 1782
        self._B0 = 12 * self.A2 / self.A0
        self.v_minp = np.sqrt(1 / (6 * self.A0) * (-self.A1 + np.sqrt(self.A1 ** 2 + 12 * self.A0 * self.A2)))
        self.v_fmax = (self.A2 / self.A0) ** (1 / 4)
        self.mode = 'mermoz_fitted'

    def power(self, asp):
        return self.A0 * asp ** 3 + self.A1 * asp + self.A2 / asp

    def d_power(self, asp):
        return 3 * self.A0 * asp ** 2 + self.A1 - self.A2 / (asp ** 2)

    def asp_opti(self, adjoint):
        pn = self._vec_or_float_to_norm(adjoint)
        a = (pn - self.A1) / self.A0
        return np.sqrt(1 / 6 * (a + np.sqrt(a ** 2 + self._B0)))
