import numpy as np
from scipy.integrate import RK45

from dabry.misc import Chrono, Utils
from dabry.problem import NavigationProblem
from dabry.trajectory import AugmentedTraj


class SolverES:

    def __init__(self, mp: NavigationProblem, mode=0, rel_tgt_thr=0.05, N_step_max=200):
        self.mp = mp
        self.mode = mode
        self.rel_tgt_thr = rel_tgt_thr
        self.N_step_max = N_step_max
        self.t_start = 0.

    def tuscale(self, t, dur=False):
        if dur:
            return t * (self.mp.model.wind.t_end - self.mp.model.wind.t_start)
        return self.mp.model.wind.t_start + t * (self.mp.model.wind.t_end - self.mp.model.wind.t_start)

    def tdscale(self, t, dur=False):
        if dur:
            return t / (self.mp.model.wind.t_end - self.mp.model.wind.t_start)
        return (t - self.mp.model.wind.t_start) / (self.mp.model.wind.t_end - self.mp.model.wind.t_start)

    def upscale(self, x):
        # xl, xu = self.mp.bl[0], self.mp.tr[0]
        # yl, yu = self.mp.bl[1], self.mp.tr[1]
        # return np.array((xl + x[0] * (xu - xl), yl + x[1] * (yu - yl)))
        return x * self.f_up

    def downscale(self, x):
        # xl, xu = self.mp.bl[0], self.mp.tr[0]
        # yl, yu = self.mp.bl[1], self.mp.tr[1]
        # return np.array(((x[0] - xl) / (xu - xl), (x[1] - yl) / (yu - yl)))
        return x * self.f_down

    def dyn(self, t, y):
        yy = np.array(y)
        yy[:2] = self.upscale(y[:2])
        return self.mp.dyn_aug(t, yy)

    def integrate(self, t_max, theta0, np0=1.):

        p0 = -np0 * np.array((np.cos(theta0), np.sin(theta0)))

        y0 = np.hstack((self.mp.x_init, p0))

        integrator = RK45(self.mp.dyn_aug, self.mp.model.wind.t_start, y0, self.mp.model.wind.t_start + t_max,
                          first_step=0.01 * t_max)
        yy = np.zeros((self.N_step_max, 4))
        i = 0
        ts = [self.mp.model.wind.t_start]
        yy[0, :] = integrator.y
        while not integrator.status == 'finished' and i < self.N_step_max:
            integrator.step()
            i += 1
            yy[i, :] = integrator.y
            ts.append(integrator.t)
            if not self.mp.domain(integrator.y[:2]):
                break
        y = np.zeros((i + 1, 4))
        y[:] = yy[:i + 1]
        return y, ts

    def integrate_precise(self, t_max, theta0, nt, np0=1.):
        p0 = -np0 * np.array((np.cos(theta0), np.sin(theta0)))
        y0 = np.hstack((self.mp.x_init, p0))
        t_start = self.mp.model.wind.t_start

        integrator = RK45(self.mp.dyn_aug, t_start, y0, t_start + t_max,
                          first_step=0.01 * t_max)
        yy = np.zeros((nt, 4))
        i = 1
        ts = np.linspace(t_start, t_start + t_max, nt)
        yy[0, :] = integrator.y
        while not integrator.status == 'finished':
            integrator.step()
            dense = integrator.dense_output()
            i0 = i
            while i < nt and ts[i] <= integrator.t:
                i += 1
            yy[i0:i, :] = dense(ts[i0:i])
        return yy, ts

    def solve(self, t_max: float, t_start=0., N_disc=200, quiet=False, dtheta=(180, 'deg')):

        self.t_start = t_start
        if len(dtheta) != 2 or dtheta[1] not in ['deg', 'rad']:
            raise Exception(
                "dtheta shall be a tuple with float value in first place and unit in ['deg', 'rad'] in second place")
        dtheta_val = dtheta[0] * (Utils.DEG_TO_RAD if dtheta[1] == 'deg' else 1.)
        theta_init = self.mp.heading(self.mp.x_init, self.mp.x_target)
        thetas = np.linspace(theta_init - dtheta_val, theta_init + dtheta_val, N_disc)

        costs = []
        trajs = []
        tts = []
        d_max = self.rel_tgt_thr * self.mp.distance(self.mp.x_init, self.mp.x_target)
        chrono = Chrono()
        if not quiet:
            chrono.start('Shooting')
        for k, theta in enumerate(thetas):
            y, ts = self.integrate(t_max, theta)
            trajs.append(y)
            tts.append(ts)
            d_min = None
            t_opt = None
            for i, d in enumerate(list(map(lambda x: self.mp.distance(self.mp.x_target, x), y[:, :2]))):
                if d_min is None or d < d_min:
                    t_opt = ts[i] - self.mp.model.wind.t_start
                    d_min = d
            costs.append((d_min, k, t_opt))
        costs = sorted(costs)
        if not quiet:
            chrono.stop()

        if costs[0][0] > d_max:
            print('No solution found')
        best_costs = []
        for k, c in enumerate(costs):
            if k == 0 or c[0] < d_max:
                best_costs.append(c)
        t_min = None
        k0 = 0
        for k, c in enumerate(best_costs):
            if t_min is None or c[2] < t_min:
                t_min = c[2]
                k0 = c[1]

        theta_opt = thetas[k0]
        y, ts = self.integrate_precise(t_min, theta_opt, 100)
        traj = AugmentedTraj(ts, y[:, :2], y[:, 2:4], np.zeros(ts.shape[0]), ts.shape[0] - 1, self.mp.coords)
        return traj
