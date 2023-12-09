from abc import ABC, abstractmethod

from numpy import ndarray

from .wind import Wind

"""
stoppingcond.py
Stopping conditions for integrator.

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


class StoppingCond(ABC):
    """
    Provides a wrapper for general integration stop conditions
    """

    @abstractmethod
    def value(self,
              t: float,
              x: ndarray) -> bool:
        """
        This method returns the truth value of the stopping condition depending on
        state and time
        :param t: Last integration timestamp
        :param x: Last integration state
        :return: True if the integrator shall stop, False else
        """
        pass


class DisjunctionSC(StoppingCond):

    def __init__(self, sc1, sc2):
        self.sc1 = sc1
        self.sc2 = sc2

    def value(self,
              t: float,
              x: ndarray) -> bool:
        return self.sc1.value(t, x) or self.sc2.value(t, x)


class MaxOneDimSC(StoppingCond):
    """
        Stopping condition on maximum value for a coordinate
    """

    def __init__(self,
                 index: int,
                 max_value: float):
        """
        :param index: The coordinate on which the conditions holds
        :param max_value: The maximum value for the selected coordinate
        """
        self.index = index
        self.max_value = max_value

    def value(self, t, x):
        return x[self.index] >= self.max_value


class TimedSC(StoppingCond):
    """
        Stopping condition on maximal time
    """

    def __init__(self, *max_times):
        """
        :param max_times: Collection of time bounds
        """
        self.max_times = []
        for t in max_times:
            self.max_times.append(t)

    def value(self, t, x):
        for tt in self.max_times:
            if t >= tt:
                return True
        return False


class PrecisionSC(StoppingCond):
    """
        Stops whenever the integration enters a zone where the characteristic
        time of the control law is not negligible compared to the integration
        step. This corresponds to zones where the integration scheme would
        not be sufficiently precize.
    """

    def __init__(self, wind: Wind, factor=1e-1, int_stepsize=1.):
        self.factor = factor
        self.int_stepsize = int_stepsize
        self.wind = wind

    def value(self, t, x):
        return 1 / self.wind.grad_norm(x) * self.factor < self.int_stepsize


class DistanceSC(StoppingCond):
    """
        Stops when trajectory is close enough to a given point according
        to a given distance function
    """

    def __init__(self, distance, ceil):
        self.distance = distance
        self.ceil = ceil

    def value(self, _, x):
        return self.distance(x) < self.ceil
