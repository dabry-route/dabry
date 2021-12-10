from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from src.trajectory import Trajectory


class Integration(ABC):

    def __init__(self,
                 wind,
                 dyn,
                 feedback,
                 stop_cond=None,
                 max_iter=10000,
                 int_step=0.0001):
        self.wind = wind
        self.dyn = dyn
        self.feedback = feedback
        self.stop_cond = stop_cond
        self.max_iter = max_iter
        self.int_step = int_step

    @abstractmethod
    def integrate(self,
                  x_init: ndarray) -> Trajectory:
        """
        Integrates the trajectory from given starting point
        :param x_init: The starting point
        :return: The integrated trajectory
        """
        pass


class IntEulerExpl(Integration):

    def integrate(self,
                  x_init):
        if self.stop_cond is None:
            raise UserWarning("No stopping condition has been specified for auto-integration")
        timestamps = np.zeros(self.max_iter)
        points = np.zeros((self.max_iter, 2))
        controls = np.zeros(self.max_iter)
        i = 0
        t = 0.
        x = np.zeros(2)
        x[:] = x_init
        points[0] = x
        controls[0] = self.feedback.value(x)
        dt = self.int_step
        a = self.stop_cond.value(t, x)
        while (i+1 < self.max_iter) and not (self.stop_cond and self.stop_cond.value(t, x)):
            i += 1
            t += dt
            u = self.feedback.value(x)
            d_val = self.dyn.value(x, u, t)
            x += dt * d_val
            timestamps[i] = t
            points[i] = x
            controls[i] = u
        return Trajectory(timestamps, points, controls, last_index=i, type="integral")
