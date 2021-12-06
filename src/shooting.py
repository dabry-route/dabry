from abc import ABC, abstractmethod

import numpy as np

from src.dynamics import Dynamics
from src.trajectory import AugmentedTraj
from wind import Wind
from numpy import ndarray


class Shooting(ABC):
    """
    Defines a shooting method for the augmented model with adjoint state
    """

    def __init__(self,
                 dyn: Dynamics,
                 x_init: ndarray,
                 final_time,
                 N_iter=1000
                 ):
        """
        :param dyn: The dynamics of the problem
        :param x_init: The initial state vector
        :param final_time: The final time for integration
        :param N_iter: The number of subdivisions for the integration scheme
        """
        self.dyn = dyn
        self.x_init = x_init
        self.final_time = final_time
        self.N_iter = N_iter
        self.p_init = np.zeros(2)

    def set_adjoint(self, p_init: ndarray):
        self.p_init[:] = p_init

    def control(self, x, p, t):
        v = -p/np.linalg.norm(p)
        res = np.arctan2(v[1], v[0])
        return res

    def integrate(self):
        """
        Integrate trajctory thanks to the shooting method with an explicit Euler scheme
        """
        if self.p_init is None:
            raise ValueError("No initial value provided for adjoint state")
        timestamps = np.zeros(self.N_iter)
        states = np.zeros((self.N_iter, 2))
        adjoints = np.zeros((self.N_iter, 2))
        controls = np.zeros(self.N_iter)
        t = 0.
        x = self.x_init
        p = self.p_init
        states[0] = x
        adjoints[0] = p
        controls[0] = self.control(x, p, 0.)
        dt = self.final_time/self.N_iter
        for i in range(1, self.N_iter):
            t += dt
            u = self.control(x, p, t)
            dyn_x = self.dyn.value(x, u, t)
            A = -self.dyn.d_value__d_state(x, u, t).transpose()
            dyn_p = A.dot(p)
            x += dt * dyn_x
            p += dt * dyn_p
            timestamps[i] = t
            states[i] = x
            adjoints[i] = p
            controls[i] = u
        return AugmentedTraj(timestamps, states, adjoints, controls, last_index=self.N_iter, type="pmp")