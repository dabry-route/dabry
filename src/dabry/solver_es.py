import numpy as np
from scipy.integrate import RK45
from scipy.interpolate import CubicHermiteSpline

from dabry.misc import Chrono, Utils
from dabry.problem import NavigationProblem
from dabry.trajectory import AugmentedTraj


class ShotInfo:

    def __init__(self, i_theta, d_min, i_min, t_opt):
        self.i_theta = i_theta
        self.d_min = d_min
        self.i_min = i_min
        self.t_opt = t_opt
        self.d_min_prec = None
        self.t_opt_prec = None


class SolverES:

    def __init__(self, mp: NavigationProblem, t_scale, mode=0, rel_tgt_thr=0.05, N_step_max=200):
        self.mp = mp
        self.mode = mode
        self.rel_tgt_thr = rel_tgt_thr
        self.N_step_max = N_step_max
        self.t_start = 0.
        self.thetas = None
        self.shot_info = []
        self.first_step_coeff = 0.05
        self.t_scale = t_scale

    def integrate(self, t_max, theta0, np0=1.):
        if self.mp.coords == Utils.COORD_CARTESIAN:
            p0 = -np0 * np.array((np.cos(theta0), np.sin(theta0)))
        else:
            # COORD_GCS
            p0 = -np0 * np.array((np.cos(np.pi / 2 - theta0), np.sin(np.pi / 2 - theta0)))

        y0 = np.hstack((self.mp.x_init, p0))

        integrator = RK45(self.mp.dyn_aug, self.mp.model.wind.t_start, y0, self.mp.model.wind.t_start + t_max,
                          first_step=self.first_step_coeff * self.t_scale)
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
        if self.mp.coords == Utils.COORD_CARTESIAN:
            p0 = -np0 * np.array((np.cos(theta0), np.sin(theta0)))
        else:
            # COORD_GCS
            p0 = -np0 * np.array((np.cos(np.pi / 2 - theta0), np.sin(np.pi / 2 - theta0)))
        y0 = np.hstack((self.mp.x_init, p0))
        t_start = self.mp.model.wind.t_start
        integrator = RK45(self.mp.dyn_aug, t_start, y0, t_start + t_max,
                          first_step=self.first_step_coeff * self.t_scale)
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
            yy[i0:i, :] = dense(ts[i0:i]).transpose()
        return yy, ts

    def precise(self, i_theta, t_max, N_sub=10):
        for k in range(i_theta - 2, i_theta + 3):
            shif = self.shot_info[k]
            i_max = shif.i_min + 1
            np0 = 1
            theta0 = self.thetas[k]
            if self.mp.coords == Utils.COORD_CARTESIAN:
                p0 = -np0 * np.array((np.cos(theta0), np.sin(theta0)))
            else:
                # COORD_GCS
                p0 = -np0 * np.array((np.cos(np.pi / 2 - theta0), np.sin(np.pi / 2 - theta0)))

            y0 = np.hstack((self.mp.x_init, p0))

            integrator = RK45(self.mp.dyn_aug, self.mp.model.wind.t_start, y0, self.mp.model.wind.t_start + t_max,
                              first_step=self.first_step_coeff * self.t_scale)
            yy = np.zeros((2 * N_sub, 4))
            i = 0
            ts = []
            yy[0, :] = integrator.y
            while i <= i_max:
                integrator.step()
                i += 1
                if i >= i_max - 1:
                    dense = integrator.dense_output()
                    if i == i_max - 1:
                        yy[:N_sub, :] = dense(np.linspace(integrator.t_old, integrator.t, 10)).transpose()
                        ts.extend(np.linspace(integrator.t_old, integrator.t, 10))
                    if i == i_max:
                        yy[N_sub:, :] = dense(np.linspace(integrator.t_old, integrator.t, 10)).transpose()
                        ts.extend(np.linspace(integrator.t_old, integrator.t, 10))
            d_min = None
            j_min = None
            for j in range(2 * N_sub):
                d = self.mp.distance(self.mp.x_target, yy[j])
                if d_min is None or d < d_min:
                    d_min = d
                    j_min = j
            self.shot_info[k].d_min_prec = d_min
            self.shot_info[k].t_opt_prec = ts[j_min] - self.mp.model.wind.t_start

    def solve(self, t_max: float, t_start=0., N_disc=50, quiet=False, dtheta=(180, 'deg'), theta_init=None,
              debug=False, info=''):

        self.t_start = t_start
        if len(dtheta) != 2 or dtheta[1] not in ['deg', 'rad']:
            raise Exception(
                "dtheta shall be a tuple with float value in first place and unit in ['deg', 'rad'] in second place")
        dtheta_val = dtheta[0] * (Utils.DEG_TO_RAD if dtheta[1] == 'deg' else 1.)
        if theta_init is None:
            theta_init = self.mp.heading(self.mp.x_init, self.mp.x_target)
        self.thetas = np.linspace(theta_init - dtheta_val, theta_init + dtheta_val, N_disc)

        costs = []
        trajs = []
        tts = []
        d_max = self.rel_tgt_thr * self.mp.distance(self.mp.x_init, self.mp.x_target)
        chrono = Chrono()
        if not quiet:
            chrono.start('Shooting')
        for i_theta, theta in enumerate(self.thetas):
            y, ts = self.integrate_precise(t_max, theta, 100)
            trajs.append(y)
            tts.append(ts)
            d_min = None
            t_opt = None
            i_min = 0
            nt = y.shape[0]
            for i in range(nt//2, nt):
                d = self.mp.distance(self.mp.x_target, y[i, :2])
                if d_min is None or d < d_min:
                    t_opt = ts[i] - self.mp.model.wind.t_start
                    d_min = d
                    i_min = i
            self.shot_info.append(ShotInfo(i_theta, d_min, i_min, t_opt))
            costs.append((d_min, i_theta))
        costs = sorted(costs)
        if not quiet:
            chrono.stop()

        if costs[0][0] > d_max:
            print('No solution found')

        i_theta_min = costs[0][1]
        #self.precise(i_theta_min, t_max)
        """
        x = self.thetas[i_theta_min - 1: i_theta_min + 2]
        y = [self.shot_info[k].d_min_prec for k in range(i_theta_min - 1, i_theta_min + 2)]
        dydx = [(self.shot_info[k + 1].d_min_prec - self.shot_info[k - 1].d_min_prec) /
                (self.thetas[k + 1] - self.thetas[k - 1]) for k in range(i_theta_min - 1, i_theta_min + 2)]

        
        try:
            spline = CubicHermiteSpline(x, y, dydx)
            r_min = 0.
            s_min = None
            for r in spline.derivative().roots():
                s = spline(r)
                if s_min is None or s < s_min:
                    r_min = r
                    s_min = s
            theta_opt = r_min
        except ValueError:
            print('Refinement failed')
            theta_opt = self.thetas[i_theta_min]
        """
        # i_min = None
        # d_min = None
        # for i in range(i_theta_min - 2, i_theta_min + 3):
        #     d = self.shot_info[i].d_min_prec
        #     if d_min is None or d < d_min:
        #         i_min = i
        #         d_min = d
        #
        # i_theta_min = i_min
        theta_opt = self.thetas[i_theta_min]

        if not debug:
            #y, ts = self.integrate_precise(self.shot_info[i_theta_min].t_opt_prec, theta_opt, 100)
            y, ts = self.integrate_precise(self.shot_info[i_theta_min].t_opt, theta_opt, 100)
            traj = AugmentedTraj(ts, y[:, :2], y[:, 2:4], np.zeros(ts.shape[0]), ts.shape[0] - 1, self.mp.coords,
                                 info=info)
            return traj, theta_opt
        else:
            trajs = []
            for i_theta, theta in enumerate(self.thetas):
                t_end = self.shot_info[i_theta].t_opt if self.shot_info[i_theta].t_opt > 1e-3 else t_max
                y, ts = self.integrate_precise(t_end, theta, 100)
                ty = Utils.TRAJ_OPTIMAL if abs(i_theta - i_theta_min) <= 2 else Utils.TRAJ_INT
                traj = AugmentedTraj(ts, y[:, :2], y[:, 2:4], np.zeros(ts.shape[0]), ts.shape[0] - 1, self.mp.coords,
                                     type=ty)
                trajs.append(traj)
            #y, ts = self.integrate_precise(self.shot_info[i_theta_min].t_opt_prec, theta_opt, 100)
            y, ts = self.integrate_precise(self.shot_info[i_theta_min].t_opt, theta_opt, 100)
            traj = AugmentedTraj(ts, y[:, :2], y[:, 2:4], np.zeros(ts.shape[0]), ts.shape[0] - 1, self.mp.coords,
                                 type=Utils.TRAJ_OPTIMAL, info='m1')
            trajs.append(traj)

            return trajs
