# from heapq import heappush, heappop
import sys

import numpy as np
from math import atan2
from shapely.geometry import Polygon, Point

from mermoz.misc import *
from mermoz.model import ZermeloGeneralModel
from mermoz.problem import MermozProblem
from mermoz.shooting import Shooting
from mermoz.trajectory import AugmentedTraj, Trajectory


def heappush(a, b):
    a.append(b[1])


def heappop(a):
    return 0, a.pop()


class FrontPoint:

    def __init__(self, identifier, tsa, i_obs=-1, ccw=True):
        """
        :param identifier: The front point unique identifier
        :param tsa: The point in (id_time, time, state, adjoint, cost) form
        """
        self.i = identifier
        self.tsa = (tsa[0], tsa[1], np.zeros(2), np.zeros(2), tsa[4])
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
        if i in self.active:
            self.active.remove(i)

    def is_active(self, i):
        return i in self.active


class CollisionBuffer:
    """
    Discretize space and keep trace of trajectories to compute collisions efficiently
    """

    def __init__(self, nx, ny, bl, tr):
        self._collbuf = [[{} for _ in range(ny)] for _ in range(nx)]
        self.bl = np.zeros(2)
        self.bl[:] = bl
        self.tr = np.zeros(2)
        self.tr[:] = tr

    def _index(self, pos):
        x, y = pos[0], pos[1]
        xx = (x - self.bl[0]) / (self.tr[0] - self.bl[0])
        yy = (y - self.bl[1]) / (self.tr[1] - self.bl[1])
        nx, ny = len(self._collbuf), len(self._collbuf[0])
        return int(nx * xx), int(ny * yy)

    def add(self, pos, id, idt):
        """
        :param pos: Position in space (x, y)
        :param id: Id of trajectory to log in
        :param idt: Time index to log in
        """
        i, j = self._index(pos)
        d = self._collbuf[i][j]
        if id not in d.keys():
            d[id] = (idt, idt)
        else:
            imin, imax = d[id]
            if idt < imin:
                imin = idt
            elif idt > imax:
                imax = idt
            d[id] = (imin, imax)

    def get_inter(self, *pos):
        """
        Get list of possible interactions with positions in pos list
        :param pos: List of 2D ndarray containing coordinates
        :return: Dictionary. Keys are trajectory identifier which could be in interaction
        with positions. Values are corresponding min and max time index for which points can be in
        interaction
        """
        res = {}
        lines = []
        cols = []
        for p in pos:
            i, j = self._index(p)
            lines.append(i)
            cols.append(j)
        # Pad between non-adjescent tiles
        imin, imax = min(lines), max(lines)
        jmin, jmax = min(cols), max(cols)
        for i in range(imin, imax + 1):
            for j in range(jmin, jmax + 1):
                try:
                    d = self._collbuf[i][j]
                    for k, l in d.items():
                        if k not in res.keys():
                            res[k] = l
                        else:
                            imin, imax = res[k]
                            if l[0] < imin:
                                imin = l[0]
                            if l[1] > imax:
                                imax = l[1]
                            res[k] = (imin, imax)
                except IndexError:
                    continue
        return res


class SolverEF:
    """
    Solver for the navigation problem using progressive extremal field computation
    """
    MODE_TIME = 0
    MODE_ENERGY = 1

    def __init__(self, mp: MermozProblem, max_time, mode=0, N_disc_init=20, rel_nb_ceil=0.05, dt=None,
                 max_steps=30, hard_obstacles=True, collbuf_shape=None, cost_ceil=None, asp_offset=0.):
        self.mp_primal = mp
        self.mp_dual = MermozProblem(**mp.__dict__)  # mp.dualize()
        mem = np.array(self.mp_dual.x_init)
        self.mp_dual.x_init = np.array(self.mp_dual.x_target)
        self.mp_dual.x_target = np.array(mem)

        self.mode = mode

        self.N_disc_init = N_disc_init
        # Neighbouring distance ceil
        self.abs_nb_ceil = rel_nb_ceil * mp._geod_l
        if dt is not None:
            self.dt = -dt
        else:
            self.dt = -.005 * mp._geod_l / mp.aero.v_minp

        # This group shall be reinitialized before new resolution
        # Contains augmented state points (id, time, state, adjoint) used to expand extremal field
        self.active = []
        self.new_points = {}
        self.p_inits = {}
        self.lsets = []
        self.rel = None
        self.n_points = 0
        self.parents_primal = {}
        self.parents_dual = {}
        self.child_order_primal = {}
        self.child_order_dual = {}
        self.child_order = None

        self.trajs_primal = {}
        self.trajs_dual = {}
        self.trajs_control = {}
        self.trajs_add_info = {}
        # Default mode is dual problem
        self.mode_primal = False
        self.mp = self.mp_dual
        self.trajs = self.trajs_dual
        self.max_extremals = 10000
        self.max_time = max_time
        self.cost_ceil = cost_ceil
        self.asp_offset = asp_offset

        self.it = 0

        self.collbuf_shape = (50, 50) if collbuf_shape is None else collbuf_shape
        self.collbuf = None

        self.reach_time = None

        self._index_p = 0
        self._index_d = 0

        self.max_steps = max_steps

        self.N_filling_steps = 1
        self.hard_obstacles = hard_obstacles

        self.debug = False

    def new_traj_index(self, par1=None, par2=None):
        if sum((par1 is None, par2 is None)) == 1:
            print('Error in new index creation : one of the two parents is None', file=sys.stderr)
            exit(1)
        if self.mode_primal:
            i = self._index_p
            self.parents_primal[i] = (par1, par2)
            self._index_p += 1
        else:
            i = self._index_d
            self.parents_dual[i] = (par1, par2)
            self._index_d += 1
        return i

    def get_parents(self, i):
        if self.mode_primal:
            p1, p2 = self.parents_primal[i]
        else:
            p1, p2 = self.parents_dual[i]
        return p1, p2

    def ancestors(self, i):
        p1, p2 = self.get_parents(i)
        if p1 is None:
            return set()
        return {p1, p2}.union(self.ancestors(p1)).union(self.ancestors(p2))

    def set_primal(self, b):
        self.mode_primal = b
        if b:
            self.mp = self.mp_primal
            self.trajs = self.trajs_primal
            self.child_order = self.child_order_primal
            if self.dt < 0:
                self.dt *= -1
        else:
            self.mp = self.mp_dual
            self.trajs = self.trajs_dual
            self.child_order = self.child_order_dual
            if self.dt > 0:
                self.dt *= -1

    def setup(self, time_offset=0.):
        """
        Push all possible angles to active list from init point
        :return: None
        """
        self.active = {}
        self.new_points = {}
        self.p_inits = {}
        self.lsets = []
        self.rel = None
        self.it = 0
        self.n_points = 0
        t_start = self.mp_primal.model.wind.t_start + time_offset
        self.collbuf = CollisionBuffer(*self.collbuf_shape, self.mp.bl, self.mp.tr)
        for k in range(self.N_disc_init):
            theta = 2 * np.pi * k / self.N_disc_init
            pd = np.array((np.cos(theta), np.sin(theta)))
            pn = 1.
            if self.mode == self.MODE_ENERGY:
                asp = self.asp_offset + self.mp.aero.asp_mlod(-pd @ self.mp.model.wind.value(t_start, self.mp.x_init))
                pn = self.mp.aero.d_power(asp)
            p = pn * pd
            i = self.new_traj_index()
            self.p_inits[i] = np.zeros(2)
            self.p_inits[i][:] = p
            self.collbuf.add(self.mp.x_init, i, 0)
            a = FrontPoint(i, (0, t_start, self.mp.x_init, p, 0.))
            self.active[i] = a
            self.trajs[k] = {}
            self.trajs[k][a.tsa[0]] = a.tsa[1:]
        self.rel = Relations(list(np.arange(self.N_disc_init)))

    def build_cfront(self):
        # Build current front
        i = self.rel.get_index()
        front = [list(self.trajs[i].values())[-1][1]]
        _, j = self.rel.get(i)
        while j != i:
            front.append(list(self.trajs[j].values())[-1][1])
            _, j = self.rel.get(j)
        return Polygon(front)

    def step_global(self):

        # Step the entire active list, look for domain violation
        active_list = []
        obstacle_list = []
        front_list = []
        for i_a, a in self.active.items():
            active_list.append(a.i)
            it = a.tsa[0]
            t, x, p, status, i_obs, ccw = self.step_single(a.tsa[1:], a.i_obs, a.ccw)
            self.n_points += 1
            # Cost computation
            dt = t - a.tsa[1]
            if self.mode == self.MODE_TIME:
                power = self.mp.aero.power(self.mp.model.v_a)
            else:
                power = self.mp.aero.power(self.mp.aero.asp_opti(p))
            c = power * dt + a.tsa[4]
            tsa = (t, np.zeros(2), np.zeros(2), c)
            tsa[1][:] = x
            tsa[2][:] = p
            self.trajs[a.i][it + 1] = tsa
            if not status and self.rel.is_active(a.i):
                self.rel.deactivate(a.i)
                obstacle_list.append(a.i)
            elif c > self.cost_ceil:
                self.rel.deactivate(a.i)
            else:
                if it == 0 or True:  # or not polyfront.contains(Point(*x)):
                    na = FrontPoint(a.i, (it + 1,) + self.trajs[a.i][it + 1], i_obs, ccw)
                    self.new_points[a.i] = na
                else:
                    self.rel.remove(a.i)
                    front_list.append(a.i)
        if self.debug:
            print(f'     Active : {tuple(active_list)}')
            print(f'   Obstacle : {tuple(obstacle_list)}')
            print(f'      Front : {tuple(front_list)}')

        for i_na, na in self.new_points.items():
            it = self.active[i_na].tsa[0]
            x_it = self.active[i_na].tsa[2]
            x_itp1 = na.tsa[2]
            interactions = self.collbuf.get_inter(x_it, x_itp1)
            stop = False
            for k, l in interactions.items():
                if k == i_na:
                    continue
                for itt in range(l[0] - 1, l[1] + 1):
                    if stop:
                        break
                    try:
                        points = x_it, x_itp1, self.trajs[k][itt][1], self.trajs[k][itt + 1][1]
                        if collision(*points):
                            _, alpha1, alpha2 = intersection(*points)
                            cost1 = (1 - alpha1) * self.active[i_na].tsa[4] + alpha1 * na.tsa[4]
                            cost2 = (1 - alpha2) * self.trajs[k][itt][3] + alpha2 * self.trajs[k][itt + 1][3]
                            if cost2 < cost1:
                                self.rel.deactivate(i_na)
                                stop = True
                    except KeyError:
                        pass
                if stop:
                    break
                p1, p2 = self.get_parents(k)
                for p in (p1, p2):
                    if p in interactions.keys():
                        l1 = interactions[k]
                        l2 = interactions[p]
                        l3 = (max(l1[0], l2[0]), min(l1[1], l2[1]))
                        for itt in range(l3[0] - 1, l3[1] + 1):
                            try:
                                if itt + 1 < it:
                                    points = x_it, x_itp1, self.trajs[k][itt][1], self.trajs[p][itt][1]
                                    if collision(*points):
                                        _, alpha1, alpha2 = intersection(*points)
                                        cost1 = (1 - alpha1) * self.active[i_na].tsa[4] + alpha1 * na.tsa[4]
                                        cost2 = (1 - alpha2) * self.trajs[k][itt][3] + alpha2 * self.trajs[p][itt][3]
                                        if cost2 < cost1:
                                            self.rel.deactivate(i_na)
                                            stop = True
                            except KeyError:
                                pass
                if stop:
                    break

        for i_na, na in self.new_points.items():
            self.collbuf.add(na.tsa[2], i_na, na.tsa[0])

        # Fill in holes when extremals get too far from one another

        for i_na in list(self.new_points.keys()):
            na = self.new_points[i_na]
            _, iu = self.rel.get(na.i)
            if self.rel.is_active(i_na) and self.rel.is_active(iu):
                # Case 1, both points are active and at least one lies out of an obstacle
                if self.mp.distance(na.tsa[2], self.new_points[iu].tsa[2]) > self.abs_nb_ceil \
                        and (na.i_obs < 0 or self.new_points[iu].i_obs < 0):
                    if self.debug:
                        print(f'Stepping between {na.i}, {iu}')
                    p1i, p2i = self.get_parents(na.i)
                    p1u, p2u = self.get_parents(iu)
                    same_parents = (p1i == p1u and p2i == p2u) or (p1i == p2u and p2i == p1u)
                    no_parents = same_parents and p1i is None
                    between_cond = na.i in (p1u, p2u) or iu in (p1i, p2i) or (same_parents and no_parents) or \
                                   (same_parents and abs(self.child_order[na.i] - self.child_order[iu]) == 1)
                    if between_cond:
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

        self.active.clear()
        for i_na, na in self.new_points.items():
            if self.rel.is_active(i_na):
                self.active[i_na] = na
        self.new_points.clear()
        self.it += 1

    def step_between(self, ill, iuu, bckw_it=0):
        it = self.it - bckw_it
        # Get samples for the adjoints as linear interpolation of angles and modulus
        pl = self.trajs[ill][it]
        pu = self.trajs[iuu][it]

        thetal = np.arctan2(*pl[2][::-1])
        thetau = np.arctan2(*pu[2][::-1])
        # thetal, thetau = DEG_TO_RAD * np.array(rectify(RAD_TO_DEG * thetal, RAD_TO_DEG * thetau))
        rhol = np.linalg.norm(pl[2])
        rhou = np.linalg.norm(pu[2])
        angles = linspace_sph(thetal, thetau, self.N_filling_steps + 2)[1:-1]
        f = 1.
        if angles.shape[0] > 1:
            if angles[1] < angles[0]:
                f = -1.
        hdgs = np.array(list(map(lambda theta: np.array((np.cos(theta), np.sin(theta))), angles)))
        rhos = f * np.linspace(rhol, rhou, self.N_filling_steps + 2)[1:-1]
        adjoints = np.einsum('i,ij->ij', rhos, hdgs)
        # TODO : write this scheme
        # Get samples for states as circle passing through borders
        # and orthogonal to lower normal

        # Get samples from linear interpolation for the states
        alpha = np.linspace(0., 1., self.N_filling_steps + 2)[1:-1]
        points = np.zeros((self.N_filling_steps, 2))
        points[:] = np.einsum('i,j->ij', 1 - alpha, pl[1]) \
                    + np.einsum('i,j->ij', alpha, pu[1])
        costs = (1 - alpha) * pl[3] + alpha * pu[3]
        new_p_inits = np.einsum('i,j->ij', 1 - alpha, self.p_inits[ill]) \
                      + np.einsum('i,j->ij', alpha, self.p_inits[iuu])
        new_indexes = [self.new_traj_index(ill, iuu) for _ in range(self.N_filling_steps)]
        for k in range(self.N_filling_steps):
            if not self.mp.domain(points[k]):
                continue
            i = new_indexes[k]
            self.child_order[i] = k
            t = pl[0]
            tsa = (t, points[k], adjoints[k], costs[k])
            self.collbuf.add(points[k], i, it)
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
                self.collbuf.add(x, i, iit)
                if not status:
                    break
                self.n_points += 1
                dt = t - self.trajs[i][iit - 1][0]
                if self.mode == self.MODE_TIME:
                    power = self.mp.aero.power(self.mp.model.v_a)
                else:
                    power = self.mp.aero.power(self.mp.aero.asp_opti(p))
                c = power * dt + self.trajs[i][iit - 1][3]
                tsa = (t, np.zeros(2), np.zeros(2), c)
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
            u = heading_opti(x, p, t, self.mp.coords)
            kw = {}
            if self.mode == self.MODE_ENERGY:
                kw['v_a'] = self.mp.aero.asp_opti(p)
            dyn_x = self.mp.model.dyn.value(x, u, t, **kw)
            A = -self.mp.model.dyn.d_value__d_state(x, u, t, **kw).transpose()
            dyn_p = A.dot(p)
            x += self.dt * dyn_x
            p += self.dt * dyn_p
            t += self.dt
            a = 1 - self.mp.model.v_a * np.linalg.norm(p) + p @ self.mp.model.wind.value(t, x)
        else:
            if obs_fllw_strategy:
                # Obstacle mode. Follow the obstacle boundary as fast as possible
                dx = self.mp._geod_l / 1e6
                phi = self.mp.phi_obs[i_obs]
                grad = np.array((1 / dx * (phi(t, x + np.array((dx, 0.))) - phi(t, x)),
                                 1 / dx * (phi(t, x + np.array((0., dx))) - phi(t, x))))
                n = grad / np.linalg.norm(grad)
                # n = self.mp.grad_phi_obs[i_obs](x) / np.linalg.norm(self.mp.grad_phi_obs[i_obs](x))
                # TODO : clarify angle definition for cartesian AND gcs
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

    def propagate(self, time_offset=0., verbose=False):
        self.setup(time_offset=time_offset)
        i = 0
        while len(self.active) != 0 and i < self.max_steps and len(self.trajs) < self.max_extremals:
            i += 1
            if verbose:
                print(i)
            self.step_global()
        msg = ''
        if i == self.max_steps:
            msg = f'Stopped on iteration limit {self.max_steps}'
        elif len(self.trajs) == self.max_extremals:
            msg = f'Stopped on extremals limit {self.max_extremals}'
        else:
            msg = f'Stopped empty active list'
        print(f'Steps : {i}, Extremals : {len(self.trajs)}, Points : {self.n_points}, {msg}')
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
        if self.mode_primal:
            self.reach_time = self.trajs[k0][iit_opt][0] - self.mp_primal.model.wind.t_start
        else:
            self.reach_time = self.mp_primal.model.wind.t_start + time_offset - self.trajs[k0][iit_opt][0]
        return self.reach_time, iit_opt, self.p_inits[k0]

    def prepare_control_law(self):
        # For the control law, keep only trajectories close to the goal
        for k, v in self.trajs_dual.items():
            m = None
            for vv in v.values():
                candidate = self.mp.distance(vv[1], self.mp_primal.x_init)
                if m is None or candidate < m:
                    m = candidate
            if m < self.mp._geod_l * 0.1:
                self.trajs_control[k] = v
                self.trajs_add_info[k] = 'control'
        # Add parent trajectories
        parents_set = set()
        for k in self.trajs_control.keys():
            parents_set = parents_set.union(self.ancestors(k))
        for kp in parents_set:
            self.trajs_control[kp] = self.trajs_dual[kp]
            self.trajs_add_info[kp] = 'control'

    def solve(self, exhaustive=False, verbose=False, no_fast=False, no_prepare_control=False, forward_only=False):
        """
        :param exhaustive: For complete coverage of state space by extremals
        :param verbose: To print steps
        :param no_fast: When problem is not time-varying, do not skip forward extremals
        :param no_prepare_control: After computation, a trajectory selection is performed unless this option
        is set to True
        :return: Minimum time to reach destination, corresponding global time index,
        corresponding initial adjoint state
        """
        if verbose: self.debug = True

        if forward_only:
            # Only compute forward front
            self.set_primal(True)
            res = self.propagate(verbose)
            polyfront = self.build_cfront()
            if not polyfront.contains(Point(*self.mp.x_target)):
                print('Warning : EF not containing target', file=sys.stderr)
                res = res[0], -1, res[2]
            return res

        elif not no_fast and self.mp.model.wind.t_end is None:
            # Problem is steady
            # Compute only backward extremals and return
            self.set_primal(False)
            res = self.propagate(verbose)
            if not no_prepare_control:
                self.prepare_control_law()
            return res
        else:
            # Problem is time-varying
            # First propagate forward extremals
            # Then compute backward extremals initialized at correct time
            # to get a closed-loop control law
            self.set_primal(True)
            self.propagate(verbose)
            polyfront = self.build_cfront()
            if not polyfront.contains(Point(*self.mp.x_target)):
                print('Warning : EF not containing target', file=sys.stderr)
            # Now than forward pass is completed, run the backward pass with correct start time
            self.set_primal(False)
            res = self.propagate(time_offset=self.reach_time, verbose=verbose)
            if not no_prepare_control:
                self.prepare_control_law()
            return res

    def control(self, t, x):
        m = None
        p = np.zeros(2)
        for k, v in self.trajs_control.items():
            for vv in v.values():
                tt = vv[0]
                if tt - t <= 10 * abs(self.dt):
                    candidate = self.mp.distance(vv[1], x)
                    if m is None or candidate < m:
                        m = candidate
                        p[:] = vv[2]
        return self.mp.control_angle(p, x)

    def aslaw(self, t, x):
        m = None
        p = np.zeros(2)
        for k, v in self.trajs_control.items():
            for vv in v.values():
                tt = vv[0]
                if tt - t <= 10 * abs(self.dt):
                    candidate = self.mp.distance(vv[1], x)
                    if m is None or candidate < m:
                        m = candidate
                        p[:] = vv[2]
        return self.mp.aero.asp_opti(p)

    def build_opti_traj(self, force_primal=False, force_dual=False):
        if force_primal or ((not force_dual) and len(self.trajs_dual) == 0):
            # Priorize dual. If empty then resort to primal.
            trajs = self.trajs_primal
            mp = self.mp_primal
            parents = self.parents_primal
            child_order = self.child_order_primal
            using_primal = True
        else:
            trajs = self.trajs_dual
            mp = self.mp_dual
            parents = self.parents_dual
            child_order = self.child_order_dual
            using_primal = False
        dist = None
        k0 = 0
        nt = 0
        s0 = 0
        for k, v in trajs.items():
            for s, kl in enumerate(v.keys()):
                i = kl
                point = v[kl][1]
                if dist is None or mp.distance(point, mp.x_target) < dist:
                    dist = mp.distance(point, mp.x_target)
                    k0 = k
                    nt = i
                    s0 = s
        points = np.zeros((nt, 2))
        ts = np.zeros(nt)
        data = list(trajs[k0].values())
        it = 0
        s = s0
        if k0 in range(self.N_disc_init):
            cond = lambda s: s > 0
        else:
            cond = lambda s: s >= 0
        while cond(s):
            points[it] = data[s][1]
            ts[it] = data[s][0]
            it += 1
            s -= 1

        if k0 not in range(self.N_disc_init):
            kl, ku = parents[k0]
            a = (child_order[k0] + 1) / (self.N_filling_steps + 1)
            while it < nt:
                endl = (nt - 1 - it) not in trajs[kl].keys()
                endu = (nt - 1 - it) not in trajs[ku].keys()
                if endl and not endu:
                    b = (child_order[kl] + 1) / (self.N_filling_steps + 1)
                    kl, _ = parents[kl]
                    a = a + b * (1 - a)
                if endu and not endl:
                    b = (child_order[ku] + 1) / (self.N_filling_steps + 1)
                    _, ku = parents[ku]
                    a = a * b
                if endl and endu:
                    b = (child_order[kl] + 1) / (self.N_filling_steps + 1)
                    c = (child_order[ku] + 1) / (self.N_filling_steps + 1)
                    kl, _ = parents[kl]
                    _, ku = parents[ku]
                    a = a * c + b * (1 - a)
                datal = [v for j, v in trajs[kl].items() if j <= (nt - 1 - it)]
                datau = [v for j, v in trajs[ku].items() if j <= (nt - 1 - it)]
                ib = min(len(datal), len(datau))
                for i in range(ib):
                    points[it] = (1 - a) * datal[-1 - i][1] + a * datau[-1 - i][1]
                    ts[it] = datal[- 1 - i][0]
                    it += 1
        if using_primal:
            ts = ts[::-1]
            points = points[::-1]
        return Trajectory(ts, points, np.zeros(nt), nt - 1, mp.coords, info=f'opt_m{self.mode}')

    def get_trajs(self, primal_only=False, dual_only=False):
        res = []
        trajgroups = []
        if not primal_only:
            trajgroups.append(self.trajs_dual)
        if not dual_only:
            trajgroups.append(self.trajs_primal)
        for i, trajs in enumerate(trajgroups):
            for it, t in trajs.items():
                n = len(t)
                timestamps = np.zeros(n)
                points = np.zeros((n, 2))
                adjoints = np.zeros((n, 2))
                controls = np.zeros(n)
                transver = np.zeros(n)
                energy = np.zeros(n)
                offset = 0.
                if self.mp_primal.model.wind.t_end is None and i == 0:
                    # If wind is steady, offset timestamps to have a <self.reach_time>-long window
                    # For dual trajectories
                    offset = 0.  # self.reach_time  # - self.max_time
                for k, e in enumerate(list(t.values())):
                    timestamps[k] = e[0] + offset
                    points[k, :] = e[1]
                    adjoints[k, :] = e[2]
                    controls[k] = self.mp.control_angle(e[2], e[1])
                    transver[k] = 1 - self.mp.model.v_a * np.linalg.norm(e[2]) + e[2] @ self.mp.model.wind.value(e[0],
                                                                                                                 e[1])
                    energy[k] = e[3]
                add_info = ''
                if i == 0:
                    try:
                        add_info = '_' + self.trajs_add_info[it]
                    except KeyError:
                        pass
                res.append(
                    AugmentedTraj(timestamps, points, adjoints, controls, last_index=n - 1, coords=self.mp.coords,
                                  label=it,
                                  type=TRAJ_PMP, info=f'ef_{i}{add_info}', transver=transver, energy=energy))

        return res
