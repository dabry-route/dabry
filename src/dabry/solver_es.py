import numpy as np
from scipy.integrate import RK45

from dabry.misc import Chrono, Utils
from dabry.problem import NavigationProblem
from dabry.trajectory import AugmentedTraj


class ShotInfo:

    def __init__(self, i_theta, theta, d_min, i_min, t_opt):
        self.i_theta = i_theta
        self.theta = theta
        self.d_min = d_min
        self.i_min = i_min
        self.t_opt = t_opt


class SolverES:

    def __init__(self, mp: NavigationProblem, max_time, max_steps=100, dt_sub=None, rel_tgt_thr=0.02):
        """
        :param mp: Problem to solve
        :param max_time: Trajectory duration upper bound
        :param max_steps: Maximum number of major steps in integration
        :param dt_sub: Subsample trajectory to given timestamp precision
        """
        self.mp = mp
        self.rel_tgt_thr = rel_tgt_thr
        self.max_steps = max_steps
        self.t_start = 0.
        self.shot_info = []
        self.first_step_coeff = 0.05
        self.max_time = max_time
        self._maxsubsteps = 201
        self.dt_sub = dt_sub if dt_sub is not None else max_time / (self._maxsubsteps - 1)
        self._index = 0
        self.trajs = []
        self.i_theta_opt = None

        self.setup()

    @property
    def theta_opt(self):
        return self.shot_info[self.i_theta_opt].theta

    @property
    def time_opt(self):
        return self.shot_info[self.i_theta_opt].t_opt

    def setup(self):
        self.trajs = []
        self.shot_info = []
        self.i_theta_opt = None

    def integrate(self, theta0, np0=1.):
        if self.mp.coords == Utils.COORD_CARTESIAN:
            p0 = -np0 * np.array((np.cos(theta0), np.sin(theta0)))
        else:
            # COORD_GCS
            p0 = -np0 * np.array((np.cos(np.pi / 2 - theta0), np.sin(np.pi / 2 - theta0)))
        y0 = np.hstack((self.mp.x_init, p0))

        integrator = RK45(self.mp.dyn_aug, self.t_start, y0, self.t_start + self.max_time,
                          first_step=self.first_step_coeff * self.max_time)
        yy = []
        i = 0
        ts = []
        t = self.t_start
        while not integrator.status == 'finished':
            integrator.step()
            dense = integrator.dense_output()
            i0 = i
            t_list = []
            while t <= integrator.t and self.mp.domain(integrator.y):
                t_list.append(t)
                i += 1
                t += self.dt_sub
            ts.extend(t_list)
            yy.extend(list(dense(np.array(t_list)).transpose()))
            if not self.mp.domain(integrator.y):
                break
        return np.column_stack((np.array(ts), np.array(yy)))

    def create_index(self):
        res = self._index
        self._index += 1
        return res

    def blast(self, theta_min, theta_max, N_disc):
        thetas = np.linspace(theta_min, theta_max, N_disc)
        for theta in thetas:
            i_theta = self.create_index()
            y = self.integrate(theta)
            self.trajs.append(y)
            d_min = None
            t_opt = None
            i_min = 0
            nt = y.shape[0]
            for i in range(0, nt):
                d = self.mp.distance(self.mp.x_target, y[i, 1:3])
                if d_min is None or d < d_min:
                    t_opt = y[i, 0] - self.mp.model.wind.t_start
                    d_min = d
                    i_min = i
            self.shot_info.append(ShotInfo(i_theta, theta, d_min, i_min, t_opt))

    def get_trajs(self, no_trunc=False):
        res = []
        for i, traj in enumerate(self.trajs):
            if self.i_theta_opt is None or no_trunc:
                i_stop = traj.shape[0]
            else:
                for i in range(traj.shape[0]):
                    if traj[i, 0] > self.t_start + self.shot_info[self.i_theta_opt].t_opt:
                        break
                i_stop = i
            ts = traj[:i_stop, 0]
            points = traj[:i_stop, 1:3]
            adjoints = traj[:i_stop, 3:5]
            traj = AugmentedTraj(ts, points, adjoints, np.zeros(i_stop), i_stop - 1, self.mp.coords,
                                 type=Utils.TRAJ_PMP, info='ef_0')
            res.append(traj)
        return res

    def solve(self, depth_max=2, N_disc=50, quiet=False, dtheta=(180, 'deg'), theta_init=None, info=''):

        self.t_start = self.mp.t_init if self.mp.t_init is not None else self.mp.model.wind.t_start
        if len(dtheta) != 2 or dtheta[1] not in ['deg', 'rad']:
            raise Exception(
                "dtheta shall be a tuple with float value in first place and unit in ['deg', 'rad'] in second place")
        dtheta_val = dtheta[0] * (Utils.DEG_TO_RAD if dtheta[1] == 'deg' else 1.)
        if theta_init is None:
            theta_init = self.mp.heading(self.mp.x_init, self.mp.x_target)

        theta_min, theta_max = theta_init - dtheta_val, theta_init + dtheta_val

        d_max = self.rel_tgt_thr * self.mp.distance(self.mp.x_init, self.mp.x_target)
        chrono = Chrono()
        if not quiet:
            chrono.start('Shooting')

        depth = 0
        found = False

        while depth < depth_max:
            i_prev = self._index
            self.blast(theta_min, theta_max, N_disc)
            i_new = self._index

            d_min = None
            for i in range(i_prev, i_new):
                if d_min is None or self.shot_info[i].d_min < d_min:
                    d_min = self.shot_info[i].d_min
                    self.i_theta_opt = self.shot_info[i].i_theta

            if d_min < d_max:
                found = True
                break
            depth += 1
            theta_min = self.shot_info[self.i_theta_opt - 1].theta
            theta_max = self.shot_info[self.i_theta_opt + 1].theta
            self.max_time = self.shot_info[self.i_theta_opt].t_opt * 1.2

        if not quiet:
            chrono.stop()

        if not found:
            print('No solution found')
            return None
        else:
            i_stop = self.shot_info[self.i_theta_opt].i_min + 1
            ts = self.trajs[self.i_theta_opt][:i_stop, 0]
            points = self.trajs[self.i_theta_opt][:i_stop, 1:3]
            adjoints = self.trajs[self.i_theta_opt][:i_stop, 3:5]
            traj = AugmentedTraj(ts, points, adjoints, np.zeros(i_stop), i_stop - 1, self.mp.coords, info=info,
                                 type=Utils.TRAJ_OPTIMAL)
            return traj
