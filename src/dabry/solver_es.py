from dabry.problem import NavigationProblem
import numpy as np
from scipy.integrate import RK45

from dabry.trajectory import AugmentedTraj
from dabry.wind import DiscreteWind


class SolverES:

    def __init__(self, mp: NavigationProblem, mode=0, rel_tgt_thr=0.05, N_step_max=200):
        self.mp = mp
        self.mode = mode
        self.rel_tgt_thr = rel_tgt_thr
        self.N_step_max = N_step_max
        self.t_start = 0.
        self.f_up = np.max((self.mp.tr - self.mp.bl))
        self.f_down = 1 / self.f_up
        self.mp.update_airspeed(
            self.mp.model.v_a * self.f_down * (self.mp.model.wind.t_end - self.mp.model.wind.t_start))
        self.mp.model.wind = self.f_down * (self.mp.model.wind.t_end - self.mp.model.wind.t_start) * self.mp.model.wind

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
        return self.mp.dyn_aug(self.tuscale(t), yy)

    def integrate(self, t_max, theta0, np0=1.):

        p0 = -np0 * np.array((np.cos(theta0), np.sin(theta0)))

        y0 = np.hstack((self.downscale(self.mp.x_init), p0))

        integrator = RK45(self.dyn, 0, y0, self.tdscale(t_max, dur=True), first_step=0.01*self.tdscale(t_max, dur=True))  # , max_step=t/(nt-1))
        yy = np.zeros((self.N_step_max, 4))
        i = 0
        ts = [0.]
        yy[0, :] = integrator.y
        while not integrator.status == 'finished' and i < self.N_step_max:
            integrator.step()
            i += 1
            yy[i, :] = integrator.y
            ts.append(integrator.t)
            if not self.mp.domain(self.upscale(integrator.y[:2])):
                break
        y = np.zeros((i + 1, 4))
        y[:] = yy[:i + 1]
        return y, np.array(list(map(self.tuscale, ts)))

    def integrate_precise(self, t_max, theta0, nt, np0=1.):
        p0 = -np0 * np.array((np.cos(theta0), np.sin(theta0)))
        y0 = np.hstack((self.downscale(self.mp.x_init), p0))

        integrator = RK45(self.dyn, 0, y0, self.tdscale(t_max, dur=True), first_step=0.01*self.tdscale(t_max, dur=True))
        yy = np.zeros((nt, 4))
        i = 1
        ts = np.linspace(0, self.tdscale(t_max, dur=True), nt)
        yy[0, :] = integrator.y
        while not integrator.status == 'finished':
            integrator.step()
            dense = integrator.dense_output()
            while i < nt and ts[i] <= integrator.t:
                yy[i, :] = dense(ts[i])
                i += 1
        return yy, np.array(list(map(self.tuscale, ts)))

    def solve(self, t_max: float, t_start=0., N_disc=200):

        self.t_start = t_start
        thetas = np.linspace(0, 2 * np.pi - 2 * np.pi / (N_disc - 1), N_disc)

        costs = []
        trajs = []
        tts = []
        d_max = self.rel_tgt_thr
        for k, theta in enumerate(thetas):
            y, ts = self.integrate(t_max, theta)
            trajs.append(y)
            tts.append(ts)
            d_min = None
            t_opt = None
            xt_scaled = self.downscale(self.mp.x_target)
            for i, d in enumerate(list(map(lambda x: np.linalg.norm(x - xt_scaled), y[:,:2]))):
                if d_min is None or d < d_min:
                    t_opt = self.tuscale(self.tdscale(ts[i]), dur=True)
                    d_min = d
            costs.append((d_min, k, t_opt))
        costs = sorted(costs)

        if costs[0][0] > d_max:
            print('No solution found')
        best_costs = []
        for k, c in enumerate(costs):
            if k==0 or c[0] < d_max:
                best_costs.append(c)
        t_min = None
        k0 = 0
        for k, c in enumerate(best_costs):
            if t_min is None or c[2] < t_min:
                t_min = c[2]
                k0 = c[1]

        theta_opt = thetas[k0]
        y, ts = self.integrate_precise(t_min, theta_opt, 100)
        traj = AugmentedTraj(ts, np.array(list(map(self.upscale, y[:,:2]))), y[:, 2:4], np.zeros(ts.shape[0]), ts.shape[0] - 1, self.mp.coords)
        return traj

