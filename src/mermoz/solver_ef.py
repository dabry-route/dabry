# from heapq import heappush, heappop
import os
from math import asin

import numpy as np
from shapely.geometry import Polygon
import csv

from mermoz.misc import *
from mermoz.problem import MermozProblem
from mermoz.trajectory import AugmentedTraj

"""
solver_ef.py
Solver for time-optimal or energy-optimal path planning problems
in a flow field using an extremal-based method (shooting method
over Pontryagin's Maximum Principle)
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


class FrontPoint:

    def __init__(self, identifier, tsa, i_obs=-1, ccw=True, rot=0.):
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
        self.rot = rot


class FrontPoint2:

    def __init__(self, idf, idt, timestamp, state, adjoint, cost, i_obs=-1, ccw=True, rot=0.):
        """
        Langrangian particle evolving through dynamics of extremals
        :param idf: Unique identifier of the trajectory to which the point belongs
        :param idt: Unique identifier of current time step
        :param timestamp: Point's timestamp
        :param state: State vector
        :param adjoint: Adjoint state vector
        :param cost: Cost of trajectory up to current point
        :param i_obs: If within an obstacle, the obstacle id (-1 if no obstacle)
        :param ccw: True if going counter clock wise wrt obstacle, False else
        :param rot: A real number capturing number of rotations around obstacle
        """
        self.idf = idf
        self.idt = idt
        self.t = timestamp
        self.state = np.array(state)
        self.adjoint = np.array(adjoint)
        self.cost = cost
        self.i_obs = i_obs
        self.ccw = ccw
        self.rot = rot


class Pareto:

    def __init__(self):
        self.durations = []
        self.energies = []
        self.filename = 'pareto.csv'

    def load(self, directory):
        filepath = os.path.join(directory, self.filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                r = csv.reader(f)
                for k, line in enumerate(r):
                    if k == 0:
                        self.durations = list(map(float, line))
                    else:
                        self.energies = list(map(float, line))
        else:
            print('Pareto : nothing to load')

    def dump(self, directory):
        with open(os.path.join(directory, self.filename), 'w') as f:
            w = csv.writer(f)
            w.writerow(self.durations)
            w.writerow(self.energies)

    def _index(self, duration):
        if duration < self.durations[0]:
            return 0
        if duration > self.durations[-1]:
            return len(self.durations) - 1
        i = 0
        j = len(self.durations) - 1
        m_idx = (i + j) // 2
        while i < j:
            m_idx = (i + j) // 2
            mid_dur = self.durations[m_idx]
            if m_idx == i or m_idx == j:
                break
            if mid_dur < duration:
                i = m_idx
            else:
                j = m_idx
        return min(i, m_idx)

    def dominated(self, x):
        """
        :param x: 2D array or len 2 Iterable containing (time, energy)
        :return: True if point is Pareto-dominated, False else
        """
        if len(self.durations) == 0:
            return False
        duration = x[0]
        energy = x[1]
        if duration < self.durations[0]:
            return False
        if duration >= self.durations[-1]:
            return energy > self.energies[-1]
        i = self._index(duration)
        # t1 = self.durations[i]
        # t2 = self.durations[i + 1]
        # e1 = self.energies[i]
        # e2 = self.energies[i + 1]
        # return e1 + (duration - t1) / (t2 - t1) * (e2 - e1) <= energy
        return duration >= self.durations[i] and energy >= self.energies[i]

    def add(self, x):
        """
        Checks if point is dominated. If not, adds it to the front. Deletes then all points
        dominated by new point.
        :param x: 2D array or len 2 Iterable containing (time, energy)
        """
        if self.dominated(x):
            return
        # Duration and energy
        t, e = x[0], x[1]
        if len(self.durations) == 0:
            self.durations.append(t)
            self.energies.append(e)
            return
        if t < self.durations[0] and e > self.energies[0]:
            self.durations = [t] + self.durations
            self.energies = [e] + self.energies
        elif t > self.durations[-1] and e < self.energies[-1]:
            self.durations = self.durations + [t]
            self.energies = self.energies + [e]
        else:
            i0 = self._index(t)
            i = i0 + 1
            for i in range(i0 + 1, len(self.durations)):
                if self.energies[i] < e:
                    break
            if i == len(self.durations) - 1 and self.energies[i] > e:
                i += 1
            # Now points from first index to last index excluded shall be removed from front
            # because they are Pareto-dominated by new point
            self.durations = self.durations[:i0 + 1] + self.durations[i:]
            self.energies = self.energies[:i0 + 1] + self.energies[i:]


class EFOptRes:

    def __init__(self, status, bests):
        """
        :param status: If problem was solved to optimality
            0: not solved, information is relative to closest trajectory
            1: solved in fast mode
            2: global optimum
        :param bests: Dictionary with keys being trajectory indexes and values containing at least the fields
            'cost': cost to reach target
            'duration': duration to reach target
            'traj': the associated trajectory
            'adjoint': the initial adjoint state that led the trajectory
        """
        self.status = status
        self.bests = bests
        self.min_cost_idx = None
        min_cost = None
        for k, v in bests.items():
            cost = v['cost']
            if min_cost is None or cost < min_cost:
                self.min_cost_idx = k
                min_cost = cost

    @property
    def cost(self):
        return self.bests[self.min_cost_idx]['cost']

    @property
    def duration(self):
        return self.bests[self.min_cost_idx]['duration']

    @property
    def index(self):
        return self.min_cost_idx

    @property
    def adjoint(self):
        res = np.zeros(2)
        res[:] = self.bests[self.min_cost_idx]['adjoint']
        return res

    @property
    def traj(self):
        return self.bests[self.min_cost_idx]['traj']


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

    def __init__(self,
                 mp: MermozProblem,
                 max_time,
                 mode=0,
                 N_disc_init=20,
                 rel_nb_ceil=0.05,
                 dt=None,
                 max_steps=None,
                 hard_obstacles=True,
                 collbuf_shape=None,
                 cost_ceil=None,
                 asp_offset=0.,
                 asp_init=None,
                 no_coll_filtering=False,
                 pareto=None):
        self.mp_primal = mp
        self.mp_dual = MermozProblem(**mp.__dict__)  # mp.dualize()
        mem = np.array(self.mp_dual.x_init)
        self.mp_dual.x_init = np.array(self.mp_dual.x_target)
        self.mp_dual.x_target = np.array(mem)

        self.mode = mode

        self.N_disc_init = N_disc_init
        # Neighbouring distance ceil
        self.abs_nb_ceil = rel_nb_ceil * mp.geod_l
        if dt is not None:
            self.dt = -dt
        else:
            if mode == 0:
                self.dt = -.005 * mp.l_ref / mp.model.v_a
            else:
                self.dt = -.005 * mp.l_ref / mp.aero.v_minp

        # This group shall be reinitialized before new resolution
        # Contains augmented state points (id, time, state, adjoint) used to expand extremal field
        self.active = {}
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
        self.asp_init = asp_init if asp_init is not None else self.mp.model.v_a

        self.it = 0

        self.collbuf_shape = (50, 50) if collbuf_shape is None else collbuf_shape
        self.collbuf = None

        self.reach_time = None
        self.reach_cost = None

        self._index_p = 0
        self._index_d = 0

        self.max_steps = max_steps if max_steps is not None else int(max_time / abs(dt))

        self.N_filling_steps = 1
        self.hard_obstacles = hard_obstacles

        # Base collision filtering scheme only for steady problems
        self.no_coll_filtering = no_coll_filtering or self.mp.model.wind.t_end is not None
        self.quick_solve = False
        self.quick_traj_idx = -1
        self.best_distances = [self.mp.geod_l]

        self.any_out_obs = False

        self.pareto = pareto

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
        self.any_out_obs = True
        t_start = self.mp_primal.model.wind.t_start + time_offset
        if not self.no_coll_filtering:
            self.collbuf = CollisionBuffer(*self.collbuf_shape, self.mp.bl, self.mp.tr)
        for k in range(self.N_disc_init):
            theta = 2 * np.pi * k / self.N_disc_init
            pd = np.array((np.cos(theta), np.sin(theta)))
            pn = 1.
            if self.mode == self.MODE_ENERGY:
                if self.asp_init is not None:
                    asp = self.asp_init
                else:
                    asp = self.asp_offset + self.mp.aero.asp_mlod(
                        -pd @ self.mp.model.wind.value(t_start, self.mp.x_init))
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
        self.any_out_obs = False
        active_list = []
        obstacle_list = []
        front_list = []
        self.best_distances = [self.mp.geod_l]
        for i_a, a in self.active.items():
            active_list.append(a.i)
            it = a.tsa[0]
            t, x, p, status, i_obs, ccw, rot = self.step_single(a.tsa[1:], a.i_obs, a.ccw, a.rot)
            if i_obs == -1:
                self.any_out_obs = True
            dist = self.mp.distance(x, self.mp.x_target)
            self.best_distances.append(dist)
            if self.quick_solve and dist < self.abs_nb_ceil:
                self.quick_traj_idx = i_a
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
            elif abs(rot) > 2 * np.pi:
                self.rel.deactivate(a.i)
            elif self.cost_ceil is not None and c > self.cost_ceil:
                self.rel.deactivate(a.i)
            elif self.pareto is not None and self.pareto.dominated((t - self.mp_primal.model.wind.t_start, c)):
                self.rel.deactivate(a.i)
            else:
                if it == 0 or True:  # or not polyfront.contains(Point(*x)):
                    na = FrontPoint(a.i, (it + 1,) + self.trajs[a.i][it + 1], i_obs, ccw, rot)
                    self.new_points[a.i] = na
                else:
                    self.rel.remove(a.i)
                    front_list.append(a.i)
        if self.debug:
            print(f'     Active : {tuple(active_list)}')
            print(f'   Obstacle : {tuple(obstacle_list)}')
            print(f'      Front : {tuple(front_list)}')

        if not self.no_coll_filtering:
            for i_na, na in self.new_points.items():
                # Deactivate new active points which fall inside front
                if self.active[i_na].i_obs >= 0:
                    # Skip active points inside obstacles
                    continue
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
                                            cost2 = (1 - alpha2) * self.trajs[k][itt][3] + alpha2 * self.trajs[p][itt][
                                                3]
                                            if cost2 < cost1:
                                                self.rel.deactivate(i_na)
                                                stop = True
                                except KeyError:
                                    pass
                    if stop:
                        break

            for i_na, na in self.new_points.items():
                if self.active[i_na].i_obs >= 0:
                    # Points in obstacles do not contribute to collision buffer
                    continue
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
            if not self.no_coll_filtering:
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
                t, x, p, status, i_obs, ccw, rot = self.step_single(self.trajs[i][iit - 1], i_obs, ccw)
                dist = self.mp.distance(x, self.mp.x_target)
                self.best_distances.append(dist)
                if self.quick_solve and dist < self.abs_nb_ceil:
                    self.quick_traj_idx = i
                if not self.no_coll_filtering:
                    self.collbuf.add(x, i, iit)
                if not status:
                    iit -= 1
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

    def step_single(self, ap, i_obs=-1, ccw=True, rot=0.):
        obs_fllw_strategy = True
        x = np.zeros(2)
        x_prev = np.zeros(2)
        p = np.zeros(2)
        x[:] = ap[1]
        x_prev[:] = ap[1]
        p[:] = ap[2]
        t = ap[0]
        u = None
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
            # For compatibility with other case
            status = True
        else:
            if obs_fllw_strategy:
                # Obstacle mode. Follow the obstacle boundary as fast as possible
                obs_grad = self.mp.obstacles[i_obs].d_value(x)

                obs_tgt = (-1. if not ccw else 1.) * np.array(((0, -1), (1, 0))) @ obs_grad

                v_a = self.mp.model.v_a

                arg_ot = atan2(*obs_tgt[::-1])
                w = self.mp.model.wind.value(t, x)
                arg_w = atan2(*w[::-1])
                norm_w = np.linalg.norm(w)
                r = norm_w / v_a * sin(arg_w - arg_ot)
                if np.abs(r) >= 1.:
                    # Impossible to follow obstacle
                    # print('Impossible !')
                    status = False
                else:
                    def gs_f(uu, va, w, reference):
                        # Ground speed projected to reference
                        return (np.array((cos(uu), sin(uu))) * va + w) @ reference

                    # Select heading that maximizes ground speed along obstacle
                    u = max([arg_ot - asin(r), arg_ot + asin(r) - pi], key=lambda uu: gs_f(uu, v_a, w, obs_tgt))
                    if self.mp.coords == COORD_GCS:
                        u = np.pi / 2. - u
                    dyn_x = self.mp.model.dyn.value(x, u, t)
                    x_prev = np.zeros(x.shape)
                    x_prev[:] = x
                    x += self.dt * dyn_x
                    p[:] = - np.array((np.cos(u), np.sin(u)))
                    t += self.dt
                    arg_ref_x = atan2(*(x - self.mp.obstacles[i_obs].ref_point)[::-1])
                    arg_ref_x_prev = atan2(*(x_prev - self.mp.obstacles[i_obs].ref_point)[::-1])
                    d_arg = arg_ref_x - arg_ref_x_prev
                    if d_arg >= 0.95 * 2 * np.pi:
                        d_arg -= 2 * np.pi
                    if d_arg <= -2 * np.pi * 0.95:
                        d_arg += 2 * np.pi
                    rot += d_arg
                    status = True
            else:
                pass

        status = status and self.mp.domain(x)
        i_obs_new = i_obs
        if status:
            obstacles = self.mp.in_obs(x)
            if i_obs in obstacles:
                obstacles.remove(i_obs)
            if len(obstacles) > 0:
                i_obs_new = obstacles.pop()
            else:
                i_obs_new = -1
            if i_obs_new != i_obs:
                if i_obs_new == -1:
                    # Keep moving in obstacle, no points leaving
                    i_obs_new = i_obs
                else:
                    if self.mp.coords == COORD_GCS:
                        u = np.pi / 2 - u
                    s = np.array((np.cos(u), np.sin(u)))
                    obs_grad = self.mp.obstacles[i_obs_new].d_value(x)
                    obs_tgt = np.array(((0, -1), (1, 0))) @ obs_grad
                    ccw = s @ obs_tgt > 0.
                    rot = 0.
        return t, x, p, status, i_obs_new, ccw, rot

    def propagate(self, time_offset=0.):
        self.setup(time_offset=time_offset)
        i = 0
        print('', end='')
        while len(self.active) != 0 \
                and i < self.max_steps \
                and len(self.trajs) < self.max_extremals \
                and ((not self.quick_solve) or self.quick_traj_idx == -1) \
                and self.any_out_obs:
            i += 1
            if self.debug:
                print(i)
            self.step_global()
            print(
                f'\rSteps : {i:>6}/{self.max_steps}, Extremals : {len(self.trajs):>6}, Active : {len(self.active):>4}, '
                f'Dist : {min(self.best_distances) / self.mp.geod_l * 100:>3.0f} ', end='', flush=True)
        if i == self.max_steps:
            msg = f'Stopped on iteration limit {self.max_steps}'
        elif len(self.trajs) == self.max_extremals:
            msg = f'Stopped on extremals limit {self.max_extremals}'
        elif self.quick_solve and self.quick_traj_idx != -1:
            msg = 'Stopped quick solve'
        else:
            msg = f'Stopped empty active list'
        print(msg)

    def prepare_control_law(self):
        # For the control law, keep only trajectories close to the goal
        for k, v in self.trajs_dual.items():
            m = None
            for vv in v.values():
                candidate = self.mp.distance(vv[1], self.mp_primal.x_init)
                if m is None or candidate < m:
                    m = candidate
            if m < self.mp.geod_l * 0.1:
                self.trajs_control[k] = v
                self.trajs_add_info[k] = 'control'
        # Add parent trajectories
        parents_set = set()
        for k in self.trajs_control.keys():
            parents_set = parents_set.union(self.ancestors(k))
        for kp in parents_set:
            self.trajs_control[kp] = self.trajs_dual[kp]
            self.trajs_add_info[kp] = 'control'

    def solve(self, quick_solve=False, verbose=False, backward=False):
        """
        :param quick_solve: Whether to stop solving when at least one extremal is sufficiently close to target
        :param verbose: To print steps
        :param backward: To run forward and backward extremal computation
        :return: Minimum time to reach destination, corresponding global time index,
        corresponding initial adjoint state
        """
        if verbose:
            self.debug = True

        self.quick_solve = quick_solve

        hello = f'{self.mp_primal.descr}'
        hello += f' | {"TIMEOPT" if self.mode == SolverEF.MODE_TIME else "ENEROPT"}'
        if self.mode == SolverEF.MODE_TIME:
            hello += f' | {self.mp_primal.model.v_a:.2f} m/s'
        hello += f' | {self.mp_primal.geod_l:.2e} m'
        nowind_time = self.mp_primal.geod_l/self.mp_primal.model.v_a
        hello += f' | scale {time_fmt(nowind_time)}'
        ortho_time = self.mp_primal.orthodromic()
        hello += f' | orthodromic {time_fmt(ortho_time) if ortho_time >= 0 else "DNR"}'
        print(hello)

        if not backward:
            # Only compute forward front
            self.set_primal(True)
            self.propagate()
            res = self.build_opti_traj()
        else:
            # First propagate forward extremals
            # Then compute backward extremals initialized at correct time
            self.set_primal(True)
            self.propagate()
            # Now than forward pass is completed, run the backward pass with correct start time
            self.set_primal(False)
            self.propagate(time_offset=self.reach_time)
            res = self.build_opti_traj()

        if res.status:
            goodbye = f'Target reached in {time_fmt(res.duration)}'
            goodbye += f' | {int(100*(res.duration/nowind_time - 1)):+d}% no wind'
            if ortho_time >= 0:
                goodbye += f' | {int(100 * (res.duration / ortho_time - 1)):+d}% orthodromic'
        else:
            goodbye = f'No solution found in time < {time_fmt(self.max_steps * abs(self.dt))}'
        print(goodbye)
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

        bests = {}
        closest = None
        dist = None
        best_cost = None
        traj_idx_closest, time_idx_closest, rel_idx_closest = 0, 0, 0
        for k, v in trajs.items():
            for s, kl in enumerate(v.keys()):
                i = kl
                duration = v[kl][0] - self.mp_primal.model.wind.t_start
                point = v[kl][1]
                cost = v[kl][3]
                cur_dist = mp.distance(point, mp.x_target)
                if dist is None or cur_dist < dist:
                    dist = cur_dist
                    traj_idx_closest = k
                    time_idx_closest = i
                    rel_idx_closest = s
                    closest = {
                        'cost': cost,
                        'duration': duration,
                        'time_idx': i,
                        'rel_idx': s,
                        'adjoint': self.p_inits[k]
                    }
                if cur_dist < self.abs_nb_ceil:
                    if best_cost is None or cost < best_cost:
                        if k not in bests.keys() or duration < bests[k]['duration']:
                            bests[k] = {
                                'cost': cost,
                                'duration': duration,
                                'time_idx': i,
                                'rel_idx': s,
                                'adjoint': self.p_inits[k]
                            }
        traj_idx_list = []
        time_idx_list = []
        rel_idx_list = []
        status = 1 if self.quick_solve else 2
        if len(bests) == 0:
            #print('Warning: Target not reached', file=sys.stderr)
            traj_idx_list.append(traj_idx_closest)
            time_idx_list.append(time_idx_closest)
            rel_idx_list.append(rel_idx_closest)
            bests = {traj_idx_closest: closest}
            status = 0
        else:
            # Filter out Pareto-dominated optima
            to_delete = []
            for a, b in bests.items():
                for aa, bb in bests.items():
                    if a == aa:
                        continue
                    if bb['cost'] < b['cost'] and bb['duration'] < b['duration']:
                        to_delete.append(a)
                        break
            for a in to_delete:
                del bests[a]
            a0 = None
            min_cost = None
            for a, b in bests.items():
                cost = b['cost']
                if min_cost is None or cost < min_cost:
                    a0 = a
                    min_cost = cost
            for a in bests.keys():
                traj_idx_list.append(a)
                time_idx_list.append(bests[a]['time_idx'])
                rel_idx_list.append(bests[a]['rel_idx'])

        for k in range(len(traj_idx_list)):
            k0 = traj_idx_list[k]
            nt = time_idx_list[k]
            s0 = rel_idx_list[k]
            points = np.zeros((nt, 2))
            adjoints = np.zeros((nt, 2))
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
                adjoints[it] = data[s][2]
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
                        adjoints[it] = (1 - a) * datal[-1 - i][2] + a * datau[-1 - i][2]
                        ts[it] = datal[- 1 - i][0]
                        it += 1
            if using_primal:
                ts = ts[::-1]
                points = points[::-1]
                adjoints = adjoints[::-1]
            traj = AugmentedTraj(ts, points, adjoints, np.zeros(nt), nt - 1, mp.coords,
                                 info=f'opt_m{self.mode}_{self.asp_init:.0f}')
            bests[k0]['traj'] = traj
        res = EFOptRes(status, bests)
        return res

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
