import warnings
from abc import ABC
from scipy.integrate import ode, odeint

from mermoz.dynamics import Dynamics
from mermoz.stoppingcond import PrecisionSC
from mermoz.trajectory import AugmentedTraj
from mermoz.misc import *


class DomainException(Exception):
    pass


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
                 factor=3e-2,
                 domain=None,
                 abort_on_precision=False,
                 fail_on_maxiter=False,
                 coords=COORD_CARTESIAN,
                 target_crit=None
                 ):
        """
        :param dyn: The dynamics of the problem
        :param x_init: The initial state vector. In cartesian, must be in meters; in GCS, must be radians
        :param final_time: The final time for integration
        :param N_iter: The number of subdivisions for fixed stepsize integration scheme or the maximum number of
        steps for adaptative integration
        :param adapt_ts: Whether to use adaptation of timestep
        :param factor: Factor to use when comparing control law characteristic
        time and integration step.
        :param abort_on_precision: Whether to abort integration when the precision is considered insufficient
        according to the local control law characteristic time
        :param fail_on_maxiter: Whether to raise and exception when iteration limit is reached in adaptative
        integration
        :param coords: COORD_CARTESIAN if working in 2D planar space, COORD_GCS if working on plane-carree projection
        of earth
        :param target_crit: If not None, shooting process will monitor optimality with this criteria while
        computing trajectories. Should take a 2D np array position vector as input and return boolean
        stating if point is close enough to goal.
        """
        self.dyn = dyn
        self.x_init = np.zeros(2)
        self.x_init[:] = x_init
        self.final_time = final_time
        self.N_iter = N_iter
        self.adapt_ts = adapt_ts
        self.factor = factor
        if not domain:
            self.domain = lambda _: True
        else:
            self.domain = domain
        self.abort_on_precision = abort_on_precision
        self.fail_on_maxiter = fail_on_maxiter
        self.p_init = np.zeros(2)
        self.coords = coords
        ensure_coords(coords)
        self.target_crit = target_crit

    def set_adjoint(self, p_init: ndarray):
        self.p_init[:] = p_init

    def integrate(self, verbose=False, custom_int=True):
        """
        Integrate trajectory thanks to the shooting method with an explicit Euler scheme
        """

        def sumup(i, t, x, p, u):
            res = ''
            res += f'Step {i}, t = {t}\n'
            res += f'x : {x}\n'
            res += f'p : {p}\n'
            res += f'u : {u}\n\n'
            return res

        if self.p_init is None:
            raise ValueError("No initial value provided for adjoint state")
        print(f'p_init : {tuple(self.p_init)}, |p_init| : {np.linalg.norm(self.p_init)}')
        if custom_int:
            timestamps = np.zeros(self.N_iter)
            states = np.zeros((self.N_iter, 2))
            adjoints = np.zeros((self.N_iter, 2))
            controls = np.zeros(self.N_iter)
            t = 0.
            x = self.x_init
            p = self.p_init
            states[0] = x
            adjoints[0] = p
            u = controls[0] = control_time_opti(x, p, 0., self.coords)
            interrupted = False
            optimal = False
            list_dt = []
            if verbose:
                print(sumup(0, t, x, p, u))
            if not self.adapt_ts:
                dt = self.final_time / self.N_iter
                sc = PrecisionSC(self.dyn.wind, factor=self.factor, int_stepsize=dt)
                i = 1
                for i in range(1, self.N_iter):
                    if self.abort_on_precision:
                        _sc_value = sc.value(t, x)
                    else:
                        _sc_value = False
                    if _sc_value or not self.domain(x):
                        interrupted = True
                        break
                    if self.target_crit is not None and self.target_crit(x):
                        interrupted = True
                        optimal = True
                        break
                    t += dt
                    u = control_time_opti(x, p, t, self.coords)
                    dyn_x = self.dyn.value(x, u, t)
                    A = -self.dyn.d_value__d_state(x, u, t).transpose()
                    dyn_p = A.dot(p)
                    x += dt * dyn_x
                    p += dt * dyn_p
                    timestamps[i] = t
                    states[i] = x
                    adjoints[i] = p
                    controls[i] = u
                    if verbose:
                        print(sumup(i, t, x, p, u))
                return AugmentedTraj(timestamps, states, adjoints, controls, i, optimal=optimal, type=TRAJ_PMP,
                                     interrupted=interrupted, coords=self.coords)
            else:
                i = 1
                while t < self.final_time and i < self.N_iter and self.domain(x):
                    dt = 1 / self.dyn.wind.grad_norm(x) * self.factor
                    t += dt
                    list_dt.append(dt)
                    u = control_time_opti(x, p, t, self.coords)
                    dyn_x = self.dyn.value(x, u, t)
                    A = -self.dyn.d_value__d_state(x, u, t).transpose()
                    dyn_p = A.dot(p)
                    x += dt * dyn_x
                    p += dt * dyn_p
                    timestamps[i] = t
                    states[i] = x
                    adjoints[i] = p
                    controls[i] = u
                    if verbose:
                        print(sumup(i, t, x, p, u))
                    i += 1
                interrupted = False
                if t < self.final_time:
                    interrupted = True
                if i == self.N_iter:
                    message = f"Adaptative integration reached step limit ({self.N_iter})"
                    if self.fail_on_maxiter:
                        raise RuntimeError(message)
                    else:
                        warnings.warn(message, stacklevel=2)
                        interrupted = True
                return AugmentedTraj(timestamps, states, adjoints, controls, last_index=i, type=TRAJ_PMP,
                                     interrupted=interrupted, coords=self.coords)

        else:
            dt = self.final_time / self.N_iter
            timestamps = np.linspace(0., self.final_time, self.N_iter)
            states = np.zeros((self.N_iter, 2))
            adjoints = np.zeros((self.N_iter, 2))
            controls = np.zeros(self.N_iter)
            global_mode = 1
            if global_mode:
                def f(z, t):
                    """
                    :param z: 4D vector concat of x and p
                    :param t: Timestamp
                    :return: Value of dynamic function
                    """
                    x = np.zeros(2)
                    x[:] = z[:2]
                    p = np.zeros(2)
                    p[:] = z[2:]
                    u = control_time_opti(x, p, t, self.coords)
                    dyn_x = self.dyn.value(x, u, t)
                    A = -self.dyn.d_value__d_state(x, u, t).transpose()
                    dyn_p = A.dot(p)
                    return np.concatenate((dyn_x, dyn_p))

                z0 = np.zeros(4)
                z0[:2] = self.x_init
                z0[2:] = self.p_init
                zz = np.array(odeint(f, z0, timestamps))
                last_index = self.N_iter
                for i in range(self.N_iter):
                    if not self.domain(zz[i, :2]):
                        last_index = i + 1
                        break
                controls = np.array(list(map(lambda z: control_time_opti(z[:2], z[2:], 0, self.coords), zz)))
                return AugmentedTraj(timestamps, zz[:, :2], zz[:, 2:], controls, last_index=last_index, type=TRAJ_PMP,
                                     interrupted=(last_index != self.N_iter), coords=self.coords)
            else:
                def f(t, z):
                    """
                    :param z: 4D vector concat of x and p
                    :param t: Timestamp
                    :return: Value of dynamic function
                    """
                    x = np.zeros(2)
                    x[:] = z[:2]
                    p = np.zeros(2)
                    p[:] = z[2:]
                    u = control_time_opti(x, p, t, self.coords)
                    dyn_x = self.dyn.value(x, u, t)
                    A = -self.dyn.d_value__d_state(x, u, t).transpose()
                    dyn_p = A.dot(p)
                    return np.concatenate((dyn_x, dyn_p))

                def solout(t, z):
                    if not self.domain(z[:2]):
                        return -1
                    return 0

                z0 = np.zeros(4)
                z0[:2] = self.x_init
                z0[2:] = self.p_init
                solver = ode(f)
                solver.set_integrator('dopri5', nsteps=self.N_iter)
                solver.set_solout(solout)
                solver.set_initial_value(z0)
                states[0, :] = z0[:2]
                adjoints[0, :] = z0[2:]
                last_index = 0
                while True:
                    z = solver.integrate(solver.t + dt)
                    if not solver.successful():
                        break
                    last_index += 1
                    states[last_index, :] = z[:2]
                    adjoints[last_index, :] = z[2:]
                    controls[last_index] = control_time_opti(z[:2], z[2:], solver.t, self.coords)
                    if last_index == self.N_iter - 1:
                        break
                controls = np.array(list(map(lambda z: control_time_opti(z[:2], z[2:], 0, self.coords), np.concatenate((states, adjoints), axis=0))))
                return AugmentedTraj(timestamps, states, adjoints, controls, last_index=self.N_iter, type=TRAJ_PMP,
                                     interrupted=False, coords=self.coords)
