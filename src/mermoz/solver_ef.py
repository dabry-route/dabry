# from heapq import heappush, heappop
import numpy as np
from math import atan2
from shapely.geometry import Polygon, Point

from mermoz.misc import *
from mermoz.model import ZermeloGeneralModel
from mermoz.problem import MermozProblem
from mermoz.shooting import Shooting
from mermoz.trajectory import AugmentedTraj



def heappush(a, b):
    a.append(b[1])


def heappop(a):
    return 0, a.pop()


class FrontPoint:

    def __init__(self, identifier, tsa, i_obs=-1, ccw=True):
        """
        :param identifier: The front point unique identifier
        :param tsa: The point in (id_time, time, state, adjoint) form
        """
        self.i = identifier
        self.tsa = (tsa[0], tsa[1], np.zeros(2), np.zeros(2))
        self.tsa[2][:] = tsa[2]
        self.tsa[3][:] = tsa[3]
        self.i_obs = i_obs
        self.ccw = ccw


class Relations:

    def __init__(self, members):
        """
        :param members: List of sorted int ids representing the initial cycle.
        Relationships go like 0 <-> 1 <-> 2 <-> ... <-> n-1 and n-1 <-> 0
        """
        self.rel = {}
        for k, m in enumerate(members):
            self.rel[m] = (members[(k - 1) % len(members)], members[(k + 1) % len(members)])
        self.active = [m for m in members]

    def linked(self, i1, i2):
        if i2 in self.rel[i1]:
            return True
        else:
            return False

    def add(self, i, il, iu):
        if not self.linked(il, iu):
            print('Trying to add member between non-linked members', file=sys.stderr)
            exit(1)
        o_ill, _ = self.rel[il]
        _, o_iuu = self.rel[iu]
        self.rel[il] = o_ill, i
        self.rel[iu] = i, o_iuu
        self.rel[i] = il, iu
        self.active.append(i)

    def remove(self, i):
        il, iu = self.rel[i]
        del self.rel[i]
        o_ill, _ = self.rel[il]
        _, o_iuu = self.rel[iu]
        self.rel[il] = o_ill, iu
        self.rel[iu] = il, o_iuu
        if self.is_active(i):
            self.active.remove(i)

    def get(self, i):
        return self.rel[i]

    def get_index(self):
        return list(self.rel.keys())[0]

    def deactivate(self, i):
        self.active.remove(i)

    def is_active(self, i):
        return i in self.active


class SolverEF:
    """
    Solver for the navigation problem using progressive extremal field computation
    """

    def __init__(self, mp: MermozProblem, N_disc_init=20, rel_nb_ceil=0.05, dt=None, max_steps=30, hard_obstacles=True):
        self.mp_primal = mp
        self.mp_dual = mp.dualize()

        self.N_disc_init = N_disc_init
        # Neighbouring distance ceil
        self.abs_nb_ceil = rel_nb_ceil * mp._geod_l
        if dt is not None:
            self.dt = -dt
        else:
            self.dt = -.005 * mp._geod_l / mp.model.v_a

        # This group shall be reinitialized before new resolution
        # Contains augmented state points (id, time, state, adjoint) used to expand extremal field
        self.active = []
        self.new_points = {}
        self.p_inits = {}
        self.lsets = []
        self.rel = None
        self.n_points = 0

        self.trajs_primal = {}
        self.trajs_dual = {}
        # Default mode is dual problem
        self.mode_primal = False
        self.mp = self.mp_dual
        self.trajs = self.trajs_dual
        self.max_extremals = 1000

        self.it = 0

        self._index_p = 0
        self._index_d = 0

        self.max_steps = max_steps

        self.N_filling_steps = 4
        self.hard_obstacles = hard_obstacles

        self.debug = False


    def new_index(self):
        if self.mode_primal:
            i = self._index_p
            self._index_p += 1
        else:
            i = self._index_d
            self._index_d += 1
        return i

    def set_primal(self, b):
        self.mode_primal = b
        if b:
            self.mp = self.mp_primal
            self.trajs = self.trajs_primal
            if self.dt < 0:
                self.dt *= -1
        else:
            self.mp = self.mp_dual
            self.trajs = self.trajs_dual
            if self.dt > 0:
                self.dt *= -1

    def setup(self):
        """
        Push all possible angles to active list from init point
        :return: None
        """
        self.active = []
        self.new_points = {}
        self.p_inits = {}
        self.lsets = []
        self.rel = None
        self.it = 0
        self.n_points = 0
        for k in range(self.N_disc_init):
            theta = 2 * np.pi * k / self.N_disc_init
            p = np.array((np.cos(theta), np.sin(theta)))
            i = self.new_index()
            self.p_inits[i] = np.zeros(2)
            self.p_inits[i][:] = p
            a = FrontPoint(i, (0, self.mp.model.wind.t_start, self.mp.x_init, p))
            heappush(self.active, (1., a))
            self.trajs[k] = {}
            self.trajs[k][a.tsa[0]] = a.tsa[1:]
        self.rel = Relations(list(np.arange(self.N_disc_init)))

    def step_global(self):

        # Build current front to test if next points fall into the latter
        i = self.rel.get_index()
        front = [list(self.trajs[i].values())[-1][1]]
        _, j = self.rel.get(i)
        while j != i:
            front.append(list(self.trajs[j].values())[-1][1])
            _, j = self.rel.get(j)
        polyfront = Polygon(front)

        # Step the entire active list, look for domain violation and front collapse
        active_list = []
        obstacle_list = []
        front_list = []
        while len(self.active) != 0:
            _, a = heappop(self.active)
            active_list.append(a.i)
            it = a.tsa[0]
            t, x, p, status, i_obs, ccw = self.step_single(a.tsa[1:], a.i_obs, a.ccw)
            self.n_points += 1
            tsa = (t, np.zeros(2), np.zeros(2))
            tsa[1][:] = x
            tsa[2][:] = p
            self.trajs[a.i][it + 1] = tsa
            if not status and self.rel.is_active(a.i):
                self.rel.deactivate(a.i)
                obstacle_list.append(a.i)
            else:
                if not polyfront.contains(Point(*x)):
                    na = FrontPoint(a.i, (it + 1,) + self.trajs[a.i][it + 1], i_obs, ccw)
                    self.new_points[a.i] = na
                else:
                    self.rel.remove(a.i)
                    front_list.append(a.i)
        if self.debug:
            print(f'     Active : {tuple(active_list)}')
            print(f'   Obstacle : {tuple(obstacle_list)}')
            print(f'      Front : {tuple(front_list)}')

        # Fill in holes when extremals get too far from one another
        for na in list(self.new_points.values()):
            _, iu = self.rel.get(na.i)
            if self.rel.is_active(iu):
                # Case 1, both points are active and at least one lies out of an obstacle
                if self.mp.distance(na.tsa[2], self.new_points[iu].tsa[2]) > self.abs_nb_ceil\
                        and (na.i_obs < 0 or self.new_points[iu].i_obs < 0):
                    if self.debug:
                        print(f'Stepping between {na.i}, {iu}')
                    self.step_between(na.i, iu)
            else:
                pass
                """
                # Case 2, one point was inactivated. It lies on the boundary whereas its neighbour is moving.
                # Try to fill in the gap up to the moving nb's timestep
                last_it = list(self.trajs[iu].keys())[-1]
                if np.linalg.norm(na.tsa[2] - self.trajs[iu][last_it][2]) > self.abs_nb_ceil:
                    self.step_between(na.i, iu, self.it - last_it)
                """
        for na in self.new_points.values():
            heappush(self.active, (1., na))
        self.new_points.clear()
        self.it += 1

    def step_between(self, ill, iuu, bckw_it=0):
        it = self.it - bckw_it
        # Get samples for the adjoints as linear interpolation of angles and modulus
        pl = self.trajs[ill][it]
        pu = self.trajs[iuu][it]

        thetal = atan2(*pl[2][::-1])
        thetau = atan2(*pu[2][::-1])
        thetal, thetau = DEG_TO_RAD * np.array(rectify(RAD_TO_DEG * thetal, RAD_TO_DEG * thetau))
        rhol = np.linalg.norm(pl[2])
        rhou = np.linalg.norm(pu[2])
        angles = np.linspace(thetal, thetau, self.N_filling_steps + 2)[1:-1]
        hdgs = np.array(list(map(lambda theta: np.array((np.cos(theta), np.sin(theta))), angles)))
        rhos = np.linspace(rhol, rhou, self.N_filling_steps + 2)[1:-1]
        adjoints = np.einsum('i,ij->ij', rhos, hdgs)
        # TODO : write this scheme
        # Get samples for states as circle passing through borders
        # and orthogonal to lower normal

        # Get samples from linear interpolation for the states
        alpha = np.linspace(0., 1., self.N_filling_steps + 2)[1:-1]
        points = np.zeros((self.N_filling_steps, 2))
        points[:] = np.einsum('i,j->ij', 1 - alpha, pl[1]) \
                    + np.einsum('i,j->ij', alpha, pu[1])
        new_p_inits = np.einsum('i,j->ij', 1 - alpha, self.p_inits[ill]) \
                    + np.einsum('i,j->ij', alpha, self.p_inits[iuu])
        new_indexes = [self.new_index() for _ in range(self.N_filling_steps)]
        for k in range(self.N_filling_steps):
            if not self.mp.domain(points[k]):
                continue
            i = new_indexes[k]
            t = pl[0]
            tsa = (t, points[k], adjoints[k])
            # if self.mp.in_obs(points[k]):
            #     continue
            self.trajs[i] = {}
            self.trajs[i][it] = tsa
            self.p_inits[i] = np.zeros(2)
            self.p_inits[i][:] = new_p_inits[k]
            iit = it
            status = True
            i_obs = -1
            ccw = True
            while iit <= self.it:
                iit += 1
                t, x, p, status, i_obs, ccw = self.step_single(self.trajs[i][iit - 1], i_obs, ccw)
                if not status:
                    break
                self.n_points += 1
                tsa = (t, np.zeros(2), np.zeros(2))
                tsa[1][:] = x
                tsa[2][:] = p
                self.trajs[i][iit] = tsa

            na = FrontPoint(i, (iit,) + self.trajs[i][iit])
            self.rel.add(i, ill if k == 0 else new_indexes[k - 1], iuu)
            if not status:
                self.rel.deactivate(i)
            self.new_points[i] = na

    def step_single(self, ap, i_obs=-1, ccw=True):
        obs_fllw_strategy = False
        x = np.zeros(2)
        p = np.zeros(2)
        x[:] = ap[1]
        p[:] = ap[2]
        t = ap[0]
        if i_obs == -1 or not self.hard_obstacles:
            # For the moment, simple Euler scheme
            u = control_time_opti(x, p, t, self.mp.coords)
            dyn_x = self.mp.model.dyn.value(x, u, t)
            A = -self.mp.model.dyn.d_value__d_state(x, u, t).transpose()
            dyn_p = A.dot(p)
            x += self.dt * dyn_x
            p += self.dt * dyn_p
            t += self.dt
        else:
            if obs_fllw_strategy:
                # Obstacle mode. Follow the obstacle boundary as fast as possible
                dx = self.mp._geod_l / 1e6
                phi = self.mp.phi_obs[i_obs]
                grad = np.array((1 / dx * (phi(t, x + np.array((dx, 0.))) - phi(t, x)),
                                        1 / dx * (phi(t, x + np.array((0., dx))) - phi(t, x))))
                n = grad / np.linalg.norm(grad)
                # n = self.mp.grad_phi_obs[i_obs](x) / np.linalg.norm(self.mp.grad_phi_obs[i_obs](x))
                theta = atan2(*n[::-1])
                u = theta + (-1. if not ccw else 1.) * acos(-self.mp.model.wind.value(t, x) @ n / self.mp.model.v_a)
                dyn_x = self.mp.model.dyn.value(x, u, t)
                x += self.dt * dyn_x
                p[:] = - np.array((np.cos(u), np.sin(u)))
                t += self.dt
            else:
                pass

        status = self.mp.domain(x)
        if status and i_obs < 0:
            i_obs = self.mp.in_obs(t, x)
            if i_obs >= 0:
                dx = self.mp._geod_l / 1e6
                phi = self.mp.phi_obs[i_obs]
                grad = np.array((1 / dx * (phi(t, x + np.array((dx, 0.))) - phi(t, x)),
                                 1 / dx * (phi(t, x + np.array((0., dx))) - phi(t, x))))
                # ccw = np.cross(self.mp.grad_phi_obs[i_obs](x), dyn_x) > 0.
                ccw = np.cross(grad, dyn_x) > 0.
        return t, x, p, status, i_obs, ccw

    def solve(self, exhaustive=False, verbose=False):
        """
        :param exhaustive: For complete coverage of state space by extremals
        :param verbose: To print steps
        :return: Minimum time to reach destination, corresponding global time index,
        corresponding initial adjoint state
        """
        if verbose: self.debug = True
        self.setup()
        i = 0
        while len(self.active) != 0 and i < self.max_steps and len(self.trajs) < self.max_extremals:
            i += 1
            if verbose:
                print(i)
            self.step_global()
        if i == self.max_steps:
            print(f'Stopped on iteration limit {self.max_steps}')
        elif len(self.trajs) == self.max_extremals:
            print(f'Stopped on extremals limit {self.max_extremals}')
        else:
            print(f'Stopped empty active list')
        print(f'Steps : {i}, Extremals : {len(self.trajs)}, Points : {self.n_points}')
        k0 = None
        m = None
        iit_opt = 0
        for k, traj in self.trajs.items():
            for iit in traj.keys():
                candidate = self.mp.distance(traj[iit][1], self.mp.x_target)
                if m is None or m > candidate:
                    m = candidate
                    k0 = k
                    iit_opt = iit
        return self.mp.model.wind.t_start - self.trajs[k0][iit_opt][0], iit_opt, self.p_inits[k0]

    def control(self, x):
        m = None
        p = np.zeros(2)
        for k, v in self.trajs_dual.items():
            for vv in v.values():
                candidate = self.mp.distance(vv[1], x)
                if m is None or candidate < m:
                    m = candidate
                    p[:] = vv[2]
        if self.mp.coords == COORD_CARTESIAN:
            return atan2(*-p[::-1])
        else:
            mat = np.diag((1/np.cos(x[1]), 1.))
            pl = mat @ p
            return np.pi / 2. - atan2(*-pl[::-1])

    def get_trajs(self, primal_only=False, dual_only=False):
        res = []
        if primal_only:
            trajs = self.trajs_primal
        elif dual_only:
            trajs = self.trajs_dual
        else:
            trajs = {**self.trajs_primal, **self.trajs_dual}
        for it, t in trajs.items():
            n = len(t)
            timestamps = np.zeros(n)
            points = np.zeros((n, 2))
            adjoints = np.zeros((n, 2))
            controls = np.zeros(n)
            for k, e in enumerate(list(t.values())):
                timestamps[k] = e[0]
                points[k, :] = e[1]
                adjoints[k, :] = e[2]
            res.append(
                AugmentedTraj(timestamps, points, adjoints, controls, last_index=n, coords=self.mp.coords, label=it, type=TRAJ_PMP))

        return res
