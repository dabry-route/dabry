from abc import ABC, abstractmethod
from numpy import ndarray


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

    def __init__(self, max_time: float):
        """
        :param max_time: The maximum time allowed
        """
        self.max_time = max_time

    def value(self, t, x):
        return t >= self.max_time
