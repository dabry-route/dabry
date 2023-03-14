from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from dabry.misc import Utils
from .trajectory import Trajectory

"""
integration.py
Basic ODE integration scheme.

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


class Integration(ABC):

    def __init__(self,
                 wind,
                 dyn,
                 feedback,
                 coords,
                 aslaw=None,
                 stop_cond=None,
                 max_iter=10000,
                 int_step=0.0001,
                 t_init=0.,
                 backward=False):
        self.wind = wind
        self.dyn = dyn
        self.feedback = feedback
        self.aslaw = aslaw
        self.stop_cond = stop_cond
        self.max_iter = max_iter
        self.int_step = int_step
        self.coords = coords
        self.t_init = t_init
        self.backward = backward

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
        t = self.t_init
        timestamps[0] = self.t_init
        x = np.zeros(2)
        x[:] = x_init
        points[0, :] = x
        val = self.feedback.value(t, x)
        if type(val) in [float, np.float64] or len(val) == 1:
            heading, asp = val, None
        else:
            heading, asp = val
        controls[0] = heading
        dt = self.int_step * (-1. if self.backward else 1.)
        if self.aslaw is None and asp is None:
            while (i + 1 < self.max_iter) and (self.stop_cond is None or not self.stop_cond.value(t, x)):
                i += 1
                t += dt
                u = self.feedback.value(t, x)
                d_val = self.dyn.value(x, u, t)
                x += dt * d_val
                timestamps[i] = t
                points[i] = x
                controls[i] = u
        else:
            while (i + 1 < self.max_iter) and (self.stop_cond is None or not self.stop_cond.value(t, x)):
                i += 1
                t += dt
                u = self.feedback.value(t, x)
                if self.aslaw is not None:
                    v_a = self.aslaw.value(t, x)
                    heading = u
                else:
                    v_a = u[1]
                    heading = u[0]
                d_val = self.dyn.value(x, heading, t, v_a=v_a)
                x += dt * d_val
                timestamps[i] = t
                points[i] = x
                controls[i] = heading
        return Trajectory(timestamps, points, controls, i, self.coords, type=Utils.TRAJ_INT)
