import warnings
from abc import ABC

import numpy as np
from numpy import ndarray

from src.dynamics import Dynamics
from src.stoppingcond import PrecisionSC
from src.trajectory import AugmentedTraj


class AdaptIntLimitWarning(Warning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Shooting(ABC):
    """
    Defines a shooting method for the augmented model with adjoint state
    """

    def __init__(self,
                 dyn: Dynamics,
                 x_init: ndarray,
                 final_time,
                 N_iter=1000,
                 adapt_ts=False,
                 ceil=1e-1,
                 domain=None,
                 stop_on_failure=False
                 ):
        """
        :param dyn: The dynamics of the problem
        :param x_init: The initial state vector
        :param final_time: The final time for integration
        :param N_iter: The number of subdivisions for fixed stepsize integration scheme or the maximum number of
        steps for adaptative integration
        :param adapt_ts: Whether to use adaptation of timestep
        :param ceil: Factor to use when comparing control law characteristic
        time and integration step.
        :param stop_on_failure: Whether to raise and exception when iteration limit is reached in adaptative
        integration
        """
        self.dyn = dyn
        self.x_init = np.zeros(2)
        self.x_init[:] = x_init
        self.final_time = final_time
        self.N_iter = N_iter
        self.adapt_ts = adapt_ts
        self.ceil = ceil
        if not domain:
            self.domain = lambda _: True
        else:
            self.domain = domain
        self.stop_on_failure = stop_on_failure
        self.p_init = np.zeros(2)

    def set_adjoint(self, p_init: ndarray):
        self.p_init[:] = p_init

    def control(self, x, p, t):
        v = -p / np.linalg.norm(p)
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
        if not self.adapt_ts:
            dt = self.final_time / self.N_iter
            sc = PrecisionSC(self.dyn.wind, ceil=self.ceil, int_stepsize=dt)
            for i in range(1, self.N_iter):
                if sc.value(t, x) or not self.domain(x):
                    break
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
        else:
            i = 1
            while t < self.final_time and i < self.N_iter and self.domain(x):
                dt = 1 / self.dyn.wind.grad_norm(x) * self.ceil
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
                i += 1
            if i == self.N_iter:
                message = f"Adaptative integration reached step limit ({self.N_iter})"
                if self.stop_on_failure:
                    raise RuntimeError(message)
                else:
                    warnings.warn(message, stacklevel=2)
            return AugmentedTraj(timestamps, states, adjoints, controls, last_index=i, type="pmp")
