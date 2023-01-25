# from heapq import heappush, heappop
import os
import threading
from math import asin

import numpy as np
from shapely.geometry import Polygon, Point, LineString
import csv
import scipy.integrate

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


class Particle:

    def __init__(self, idf, idt, t, state, adjoint, cost, i_obs=-1, ccw=True, rot=0.):
        """
        Langrangian particle evolving through dynamics of extremals
        :param idf: Unique identifier of the trajectory to which the point belongs
        :param idt: Unique identifier of current time step
        :param t: Point's timestamp
        :param state: State vector
        :param adjoint: Adjoint state vector
        :param cost: Cost of trajectory up to current point
        :param i_obs: If within an obstacle, the obstacle id (-1 if no obstacle)
        :param ccw: True if going counter clock wise wrt obstacle, False else
        :param rot: A real number capturing number of rotations around obstacle
        """
        self.idf = idf
        self.idt = idt
        self.t = t
        self.state = np.array(state)
        self.adjoint = np.array(adjoint)
        self.cost = cost
        self.i_obs = i_obs
        self.ccw = ccw
        self.rot = rot


class PartialTraj:

    def __init__(self, id_start, particles):
        self.id_traj = particles[0].idf
        if len(particles) > 0:
            # Check consistency
            assert (particles[0].idt == id_start)
            for i in range(len(particles) - 1):
                assert (particles[i + 1].idt - particles[i].idt == 1)
                assert (particles[i + 1].idf == self.id_traj)
        self.id_start = id_start
        self.particles = particles

    def append(self, p: Particle):
        # assert (p.idt - self.points[-1].idt == 1)
        assert (p.idf == self.id_traj)
        self.particles.append(p)

    def to_traj(self, coords):
        n = len(self.particles)
        ts = np.zeros(n)
        states = np.zeros((n, 2))
        adjoints = np.zeros((n, 2))
        for i, pcl in enumerate(self.particles):
            ts[i] = pcl.t
            states[i, :] = pcl.state
            adjoints[i, :] = pcl.adjoint
        return AugmentedTraj(ts, states, adjoints, np.zeros(n), n - 1, coords)

    def __getitem__(self, item):
        return self.particles[item - self.id_start]

    def get_last(self):
        return self.particles[-1]


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

    def __init__(self, status, bests, trajs, mp):
        """
        :param status: If problem was solved to optimality
            0: not solved, information is relative to closest trajectory
            1: solved in fast mode
            2: global optimum
        :param bests: Dictionary with keys being trajectory indexes and values being Particles reaching target
        with optimal cost
        :param trajs: Dictionary with keys being trajectory indexes and values being optimal trajectories to
        which optimal particles belong
        :param mp: MermozProblem
        """
        self.status = status
        self.bests = bests
        self.trajs = trajs
        self.mp = mp
        self.min_cost_idx = sorted(bests, key=lambda k: bests[k].cost)[0]

    @property
    def cost(self):
        return self.bests[self.min_cost_idx].cost

    @property
    def duration(self):
        return self.bests[self.min_cost_idx].t - self.mp.model.wind.t_start

    @property
    def index(self):
        return self.min_cost_idx

    @property
    def adjoint(self):
        res = np.zeros(2)
        res[:] = self.bests[self.min_cost_idx].adjoint
        return res

    @property
    def traj(self):
        return self.trajs[self.min_cost_idx]


class Relations:

    def __init__(self):

        self.rel = {}
        self.active = []
        self.captive = []
        self.deact_reason = {}
        # Store only index of left-hand member of dead relation
        self.dead_links = set()

    def fill(self, members):
        """
        :param members: List of sorted int ids representing the initial cycle.
        Relationships go like 0 <-> 1 <-> 2 <-> ... <-> n-1 and n-1 <-> 0
        """
        for k, m in enumerate(members):
            self.rel[m] = (members[(k - 1) % len(members)], members[(k + 1) % len(members)])
        self.active = [m for m in members]

    def linked(self, i1, i2):
        if i2 in self.rel[i1]:
            return True
        else:
            return False

    def add(self, i, il, iu, force=False):
        if not self.linked(il, iu) and not force:
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

    def deactivate(self, i, reason=''):
        if i in self.active:
            self.active.remove(i)
            self.deact_reason[i] = reason

    def set_captive(self, i):
        self.captive.append(i)

    def deact_link(self, i1, i2):
        i1l, i1u = self.get(i1)
        if i2 == i1u:
            self.dead_links.add(i1)
        else:
            self.dead_links.add(i2)

    def is_active(self, i):
        return i in self.active

    def components(self, free_only=False):
        comp = [[]]
        icomp = 0
        if len(self.active) == 0:
            return []
        i = i0 = self.active[0]
        k = 1
        # while i in self.dead_links:
        #     i = i0 = self.active[k]
        #     k += 1
        i_active = True
        i_captive = False
        comp[icomp].append(i)
        _, j = self.get(i)
        in_comp = True
        start = True
        #print(f'\n{i0}')
        i1, i2 = self.get(i0)
        # print(i1, i2)
        # print(self.get(i1))
        # print(self.get(i2))
        s = set()
        s2 = set()
        while i != i0 or start:
            s.add(j)
            if s == s2:
                # print(s)
                # print(j)
                # print(self.get(j))
                raise Exception('')
            start = False
            j_active = j in self.active
            j_captive = j in self.captive
            if i_active and j_active and (i not in self.dead_links) and \
                    ((not free_only) or (not i_captive and not j_captive)):
                if not in_comp:
                    icomp += 1
                    comp.append([])
                comp[icomp].append(j)
                in_comp = True
            else:
                in_comp = False
            i_active = j_active
            i_captive = j_captive
            i = j
            s2.add(j)
            _, j = self.get(i)
        return comp


class CollisionBuffer:
    """
    Discretize space and keep trace of trajectories to compute collisions efficiently
    """

    def __init__(self):
        self._collbuf = [[]]
        self.bl = np.zeros(2)
        self.tr = np.zeros(2)

    def init(self, nx, ny, bl, tr):
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

    def add_particle(self, p: Particle):
        self.add(p.state, p.idf, p.idt)

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
                 collbuf_shape=None,
                 cost_ceil=None,
                 asp_offset=0.,
                 asp_init=None,
                 no_coll_filtering=False,
                 quick_solve=False,
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
        self.active = []
        self.new_points = []
        self.p_inits = {}
        self.lsets = []
        self.rel = Relations()
        self.n_points = 0
        self.parents_primal = {}
        self.parents_dual = {}

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
        self.collbuf = CollisionBuffer()

        self.reach_time = None
        self.reach_cost = None

        self._index_p = 0
        self._index_d = 0

        self.max_steps = max_steps if max_steps is not None else int(max_time / abs(self.dt))

        self.N_filling_steps = 1

        # Base collision filtering scheme only for steady problems
        self.no_coll_filtering = no_coll_filtering or self.mp.model.wind.t_end is not None
        self.quick_solve = quick_solve
        self.quick_traj_idx = -1
        self.best_distances = [self.mp.geod_l]

        self.any_out_obs = False

        self.manual_integration = True

        self.step_algo = 0

        self.pareto = pareto

        self.trimming_rate = 10

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
            if self.dt < 0:
                self.dt *= -1
        else:
            self.mp = self.mp_dual
            self.trajs = self.trajs_dual
            if self.dt > 0:
                self.dt *= -1

    def setup(self, time_offset=0.):
        """
        Push all possible angles to active list from init point
        :return: None
        """
        self.active = []
        self.new_points = []
        self.p_inits = {}
        self.lsets = []
        self.rel = Relations()
        self.it = 0
        self.n_points = 0
        self.any_out_obs = True
        t_start = self.mp_primal.model.wind.t_start + time_offset
        if not self.no_coll_filtering:
            self.collbuf.init(*self.collbuf_shape, self.mp.bl, self.mp.tr)
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
            pcl = Particle(i, 0, t_start, self.mp.x_init, p, 0.)
            if not self.no_coll_filtering:
                self.collbuf.add_particle(pcl)
            self.active.append(pcl)
            self.save(pcl)
        self.rel.fill(list(np.arange(self.N_disc_init)))

    def save(self, p: Particle):
        id_traj = p.idf
        if id_traj not in self.trajs.keys():
            self.trajs[id_traj] = PartialTraj(p.idt, [p])
        else:
            self.trajs[id_traj].append(p)

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

        self.any_out_obs = False
        self.best_distances = [self.mp.geod_l]

        # Evolve active particles
        for pcl in self.active:
            idf = pcl.idf

            pcl_new, status = self.step_single(pcl)

            if pcl_new.i_obs == -1:
                self.any_out_obs = True
            elif pcl.i_obs == -1:
                self.rel.set_captive(idf)

            dist = self.mp.distance(pcl_new.state, self.mp.x_target)
            self.best_distances.append(dist)
            if self.quick_solve and dist < self.abs_nb_ceil:
                self.quick_traj_idx = idf
            if not status:
                self.rel.deactivate(idf, 'domain')
            elif abs(pcl_new.rot) > 2 * np.pi:
                self.rel.deactivate(idf, 'roundabout')
            elif self.cost_ceil is not None and pcl_new.cost > self.cost_ceil:
                self.rel.deactivate(idf, 'cost')
            elif self.pareto is not None and \
                    self.pareto.dominated((pcl_new.t - self.mp_primal.model.wind.t_start, pcl_new.cost)):
                self.rel.deactivate(idf, 'pareto')
            else:
                self.new_points.append(pcl_new)
                self.save(pcl_new)
                self.n_points += 1

        # Collision filtering if needed
        if not self.no_coll_filtering:
            self.collision_filter()

            for pcl in self.new_points:
                if pcl.i_obs >= 0:
                    # Points in obstacles do not contribute to collision buffer
                    continue
                self.collbuf.add_particle(pcl)

        # Resampling of the front
        for pcl in self.new_points:
            idf = pcl.idf
            _, iu = self.rel.get(idf)
            # Resample only between two active trajectories
            if not (self.rel.is_active(idf) and self.rel.is_active(iu)):
                continue

            pcl_other = self.trajs[iu][pcl.idt]
            # Resample when precision is lost
            if self.mp.distance(pcl.state, pcl_other.state) <= self.abs_nb_ceil:
                continue
            # Resample when at least one point lies outside of an obstacle
            if pcl.i_obs >= 0 and pcl_other.i_obs >= 0:
                self.rel.deact_link(pcl.idf, pcl_other.idf)
                continue

            p1i, p2i = self.get_parents(idf)
            p1u, p2u = self.get_parents(iu)
            same_parents = (p1i == p1u and p2i == p2u) or (p1i == p2u and p2i == p1u)
            no_parents = same_parents and p1i is None
            between_cond = idf in (p1u, p2u) or iu in (p1i, p2i) or (same_parents and no_parents) or same_parents
            if not between_cond:
                continue

            # Resampling needed
            self.create_between(idf, iu, pcl.idt)

        # Trimming procedure
        if self.no_coll_filtering and self.mode == 0:
            if self.it % self.trimming_rate == 0:
                self.trim()

        self.active.clear()
        for pcl in self.new_points:
            if self.rel.is_active(pcl.idf):
                self.active.append(pcl)
        self.new_points.clear()
        self.it += 1

    def create_between(self, i1, i2, idt):
        """
        Creates new trajectory between given ones at given time index
        :param i1: First traj. index
        :param i2: Second traj. index
        :param idt: Time index
        :return: a new Particle
        """
        pcl1 = self.trajs[i1][idt]
        pcl2 = self.trajs[i2][idt]

        thetal = np.arctan2(*pcl1.adjoint[::-1])
        thetau = np.arctan2(*pcl2.adjoint[::-1])
        # thetal, thetau = DEG_TO_RAD * np.array(rectify(RAD_TO_DEG * thetal, RAD_TO_DEG * thetau))
        rhol = np.linalg.norm(pcl1.adjoint)
        rhou = np.linalg.norm(pcl2.adjoint)
        angles = linspace_sph(thetal, thetau, 3)
        f = 1.
        if angles.shape[0] > 1:
            if angles[1] < angles[0]:
                f = -1.
        theta = angles[1]
        hdg = np.array((np.cos(theta), np.sin(theta)))
        rho = f * 0.5 * (rhol + rhou)
        adjoint = rho * hdg

        state = 0.5 * (pcl1.state + pcl2.state)
        cost = 0.5 * (pcl1.cost + pcl2.cost)
        new_p_init = 0.5 * (self.p_inits[i1] + self.p_inits[i2])
        i_new = self.new_traj_index(i1, i2)
        self.p_inits[i_new] = np.zeros(2)
        self.p_inits[i_new][:] = new_p_init

        pcl_new = Particle(i_new, idt, pcl1.t, state, adjoint, cost)
        self.new_points.append(pcl_new)
        if not self.no_coll_filtering:
            self.collbuf.add_particle(pcl_new)
        self.rel.add(i_new, i1, i2)
        self.save(pcl_new)

    def step_single(self, pcl: Particle):
        x = np.zeros(2)
        x_prev = np.zeros(2)
        p = np.zeros(2)
        x[:] = pcl.state
        x_prev[:] = pcl.state
        p[:] = pcl.adjoint
        t = pcl.t
        u = None
        ccw = pcl.ccw
        rot = pcl.rot
        if pcl.i_obs == -1:
            if self.manual_integration:
                # For the moment, simple Euler scheme
                u = self.mp.control_angle(p, x)
                kw = {}
                if self.mode == self.MODE_ENERGY:
                    kw['v_a'] = self.mp.aero.asp_opti(p)
                dyn_x = self.mp.model.dyn.value(x, u, t, **kw)
                A = -self.mp.model.dyn.d_value__d_state(x, u, t, **kw).transpose()
                dyn_p = A.dot(p)
                x += self.dt * dyn_x
                p += self.dt * dyn_p
                t += self.dt
                # Transversality condition for debug purposes
                a = 1 - self.mp.model.v_a * np.linalg.norm(p) + p @ self.mp.model.wind.value(t, x)
                # For compatibility with other case
                status = True
            else:
                def f(t, z):
                    x = z[:2]
                    p = z[2:]
                    u = self.mp.control_angle(p, x)
                    kw = {}
                    if self.mode == self.MODE_ENERGY:
                        kw['v_a'] = self.mp.aero.asp_opti(p)
                    dyn_x = self.mp.model.dyn.value(x, u, t, **kw)
                    A = -self.mp.model.dyn.d_value__d_state(x, u, t, **kw).transpose()
                    dyn_p = A.dot(p)
                    return np.hstack((dyn_x, dyn_p))

                sol = scipy.integrate.solve_ivp(f, [pcl.t, pcl.t + self.dt], np.hstack((x, p)))
                x = sol.y[:2, -1]
                p = sol.y[2:, -1]
                t += self.dt
                status = True
        else:
            # Obstacle mode. Follow the obstacle boundary as fast as possible
            obs_grad = self.mp.obstacles[pcl.i_obs].d_value(x)

            obs_tgt = (-1. if not pcl.ccw else 1.) * np.array(((0, -1), (1, 0))) @ obs_grad

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
                arg_ref_x = atan2(*(x - self.mp.obstacles[pcl.i_obs].ref_point)[::-1])
                arg_ref_x_prev = atan2(*(x_prev - self.mp.obstacles[pcl.i_obs].ref_point)[::-1])
                d_arg = arg_ref_x - arg_ref_x_prev
                if d_arg >= 0.95 * 2 * np.pi:
                    d_arg -= 2 * np.pi
                if d_arg <= -2 * np.pi * 0.95:
                    d_arg += 2 * np.pi
                pcl.rot += d_arg
                status = True

        status = status and self.mp.domain(x)
        i_obs_new = pcl.i_obs
        if status:
            obstacles = self.mp.in_obs(x)
            if pcl.i_obs in obstacles:
                obstacles.remove(pcl.i_obs)
            if len(obstacles) > 0:
                i_obs_new = obstacles.pop()
            else:
                i_obs_new = -1
            if i_obs_new != pcl.i_obs:
                if i_obs_new == -1:
                    # Keep moving in obstacle, no points leaving
                    i_obs_new = pcl.i_obs
                else:
                    if self.mp.coords == COORD_GCS:
                        u = np.pi / 2 - u
                    s = np.array((np.cos(u), np.sin(u)))
                    obs_grad = self.mp.obstacles[i_obs_new].d_value(x)
                    obs_tgt = np.array(((0, -1), (1, 0))) @ obs_grad
                    ccw = s @ obs_tgt > 0.
                    rot = 0.
        if self.mode == self.MODE_TIME:
            power = self.mp.aero.power(self.mp.model.v_a)
        else:
            power = self.mp.aero.power(self.mp.aero.asp_opti(p))
        cost = power * abs(self.dt) + pcl.cost
        new_pcl = Particle(pcl.idf, pcl.idt + 1, t, x, p, cost, i_obs_new, ccw, rot)
        return new_pcl, status

    def collision_filter(self):
        for pcl in self.new_points:
            idf = pcl.idf
            # Deactivate new active points which fall inside front
            if pcl.i_obs >= 0:
                # Skip active points inside obstacles
                continue
            it = pcl.idt
            pcl_prev = self.trajs[idf][pcl.idt - 1]
            x_it = pcl_prev.state
            x_itp1 = pcl.state
            interactions = self.collbuf.get_inter(x_it, x_itp1)
            stop = False
            for k, l in interactions.items():
                if k == idf:
                    continue
                for itt in range(l[0], l[1]):
                    if stop:
                        break
                    pcl1 = self.trajs[k][itt]
                    pcl2 = self.trajs[k][itt + 1]
                    points = x_it, x_itp1, pcl1.state, pcl2.state
                    if has_intersec(*points):
                        _, alpha1, alpha2 = intersection(*points)
                        cost1 = (1 - alpha1) * pcl_prev.cost + alpha1 * pcl.cost
                        cost2 = (1 - alpha2) * pcl1.cost + alpha2 * pcl2.cost
                        if cost2 < cost1:
                            self.rel.deactivate(idf, 'collbuf')
                            stop = True
                if stop:
                    break
                p1, p2 = self.get_parents(k)
                for p in (p1, p2):
                    if p not in interactions.keys():
                        continue
                    if p == idf:
                        continue
                    l1 = interactions[k]
                    l2 = interactions[p]
                    l3 = (max(l1[0], l2[0]), min(l1[1], l2[1]))
                    for itt in range(l3[0], l3[1] + 1):
                        if itt + 1 >= it:
                            break
                        pcl1 = self.trajs[k][itt]
                        pcl2 = self.trajs[p][itt]
                        points = x_it, x_itp1, pcl1.state, pcl2.state
                        if has_intersec(*points):
                            _, alpha1, alpha2 = intersection(*points)
                            cost1 = (1 - alpha1) * pcl_prev.cost + alpha1 * pcl.cost
                            cost2 = (1 - alpha2) * pcl1.cost + alpha2 * pcl2.cost
                            if cost2 < cost1:
                                self.rel.deactivate(idf, 'collbuf_par')
                                stop = True
                if stop:
                    break

    def trim(self):
        comps = self.rel.components(free_only=True)
        for comp in comps:
            if len(comp) == 0:
                continue
            cyclic = comp[0] == comp[-1]
            n_edges = len(comp) - 1
            intersec = []
            for i in range(n_edges):
                for j in range(i + 2, n_edges):
                    if cyclic and i == 0 and j == n_edges - 1:
                        continue
                    pcl1 = self.trajs[comp[i]].get_last()
                    pcl2 = self.trajs[comp[i + 1]].get_last()
                    pcl3 = self.trajs[comp[j]].get_last()
                    pcl4 = self.trajs[comp[j + 1]].get_last()
                    if has_intersec(pcl1.state, pcl2.state, pcl3.state, pcl4.state):
                        s, _, _ = intersection(pcl1.state, pcl2.state, pcl3.state, pcl4.state)
                        v1 = 1 / self.mp.distance(pcl1.state, s)
                        v2 = 1 / self.mp.distance(pcl4.state, s)
                        alpha = v1 / (v1 + v2)
                        state = alpha * pcl1.state + (1 - alpha) * pcl4.state
                        adjoint = alpha * pcl1.adjoint + (1 - alpha) * pcl4.adjoint
                        cost = alpha * pcl1.cost + (1 - alpha) * pcl4.cost

                        idf = self.new_traj_index(comp[i], comp[j + 1])
                        new_p_init = 0.5 * (self.p_inits[comp[i]] + self.p_inits[comp[j + 1]])
                        self.p_inits[idf] = np.zeros(2)
                        self.p_inits[idf][:] = new_p_init
                        pcl = Particle(idf, pcl1.idt, pcl1.t, state, adjoint, cost)
                        intersec.append((i, j, pcl))

            def clen(i, j):
                if j < i:
                    j += n_edges
                return j - i

            if len(comps) == 1:
                intersec = sorted(intersec, key=lambda e: clen(e[0], e[1]))[:-1]
            for i, j, pcl in intersec:
                for k in range(i + 1, j + 1):
                    self.rel.deactivate(comp[k], reason='trimming')
                self.new_points.append(pcl)
                self.rel.add(pcl.idf, comp[i], comp[j + 1], force=True)
                self.save(pcl)

    def exit_cond(self):
        return len(self.active) == 0 or \
               self.step_algo >= self.max_steps or \
               len(self.trajs) >= self.max_extremals or \
               (self.quick_solve and self.quick_traj_idx != -1) or \
               (not self.any_out_obs)

    def propagate(self, time_offset=0., verbose=2):
        self.setup(time_offset=time_offset)
        self.step_algo = 0
        print('', end='')

        while not self.exit_cond():
            self.step_algo += 1
            self.step_global()
            if verbose == 2:
                print(
                    f'\rSteps : {self.step_algo:>6}/{self.max_steps}, '''
                    f'Extremals : {len(self.trajs):>6}, Active : {len(self.active):>4}, '
                    f'Dist : {min(self.best_distances) / self.mp.geod_l * 100:>3.0f} '
                    f'Comp : {len(self.rel.components(free_only=True))} ', end='', flush=True)
        if self.step_algo == self.max_steps:
            msg = f'Stopped on iteration limit {self.max_steps}'
        elif len(self.trajs) == self.max_extremals:
            msg = f'Stopped on extremals limit {self.max_extremals}'
        elif self.quick_solve and self.quick_traj_idx != -1:
            msg = 'Stopped quick solve'
        else:
            msg = f'Stopped empty active list'
        if verbose == 2:
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

    def solve(self, verbose=2, backward=False):
        """
        :param verbose: Verbosity level. 2 is common, 1 for multithread, 0 for none.
        :param backward: To run forward and backward extremal computation
        :return: Minimum time to reach destination, corresponding global time index,
        corresponding initial adjoint state
        """

        hello = f'{self.mp_primal.descr}'
        hello += f' | {"TIMEOPT" if self.mode == SolverEF.MODE_TIME else "ENEROPT"}'
        if self.mode == SolverEF.MODE_TIME:
            hello += f' | {self.mp_primal.model.v_a:.2f} m/s'
        hello += f' | {self.mp_primal.geod_l:.2e} m'
        nowind_time = self.mp_primal.geod_l / self.mp_primal.model.v_a
        hello += f' | scale {time_fmt(nowind_time)}'
        ortho_time = self.mp_primal.orthodromic()
        hello += f' | orthodromic {time_fmt(ortho_time) if ortho_time >= 0 else "DNR"}'
        if verbose == 2:
            print(hello)

        chrono = Chrono(no_verbose=True)
        chrono.start()

        if not backward:
            # Only compute forward front
            self.set_primal(True)
            self.propagate(verbose=verbose)
            res = self.build_opti_traj()
        else:
            # First propagate forward extremals
            # Then compute backward extremals initialized at correct time
            self.set_primal(True)
            self.propagate()
            # Now than forward pass is completed, run the backward pass with correct start time
            self.set_primal(False)
            self.propagate(time_offset=self.reach_time, verbose=verbose)
            res = self.build_opti_traj()

        chrono.stop()

        if res.status:
            goodbye = f'Target reached in {time_fmt(res.duration)}'
            goodbye += f' | {int(100 * (res.duration / nowind_time - 1)):+d}% no wind'
            if ortho_time >= 0:
                goodbye += f' | {int(100 * (res.duration / ortho_time - 1)):+d}% orthodromic'
            goodbye += f' | cpu time {chrono}'
        else:
            goodbye = f'No solution found in time < {time_fmt(self.max_steps * abs(self.dt))}'
        if verbose == 2:
            print(goodbye)
        elif verbose == 1:
            print(hello + '\n ↳ ' + goodbye)
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
            using_primal = True
        else:
            trajs = self.trajs_dual
            mp = self.mp_dual
            parents = self.parents_dual
            using_primal = False

        bests = {}
        btrajs = {}
        closest = None
        dist = None
        best_cost = None
        for k, ptraj in trajs.items():
            for pcl in ptraj.particles:
                duration = pcl.t - self.mp_primal.model.wind.t_start
                cur_dist = mp.distance(pcl.state, mp.x_target)
                if dist is None or cur_dist < dist:
                    dist = cur_dist
                    closest = pcl
                if cur_dist < self.abs_nb_ceil:
                    if best_cost is None or pcl.cost < best_cost:
                        if k not in list(map(lambda x: x.idf, bests.values())) or \
                                duration < bests[k].t - self.mp_primal.model.wind.t_start:
                            best_cost = pcl.cost
                            bests[k] = pcl

        status = 1 if self.quick_solve else 2
        if len(bests) == 0:
            # print('Warning: Target not reached', file=sys.stderr)
            bests = {closest.idf: closest}
            status = 0
        else:
            # Filter out Pareto-dominated optima
            dominated = []
            for k1, b1 in bests.items():
                if k1 in dominated:
                    continue
                for k2 in range(k1 + 1, len(bests.keys())):
                    if k2 in dominated:
                        continue
                    b2 = bests[k2]
                    if b2.cost < b1.cost and b2.t < b1.t:
                        dominated.append(k1)
                        break
                    if b1.cost < b2.cost and b1.t < b2.t:
                        dominated.append(k2)
            for k in dominated:
                del bests[k]

        # Build optimal trajectories
        for pcl in bests.values():
            nt = pcl.idt + 1
            k0 = pcl.idf
            p_traj = self.trajs[k0]
            points = np.zeros((nt, 2))
            adjoints = np.zeros((nt, 2))
            controls = np.zeros(nt)
            ts = np.zeros(nt)
            for s in range(p_traj.id_start, pcl.idt + 1):
                ts[s] = p_traj[s].t
                points[s] = p_traj[s].state
                adjoints[s] = p_traj[s].adjoint
                controls[s] = self.mp.control_angle(p_traj[s].adjoint, p_traj[s].state)

            sm = p_traj.id_start - 1

            if k0 not in range(self.N_disc_init):
                kl, ku = parents[k0]
                a = 0.5
                while sm >= 0:
                    endl = sm < trajs[kl].id_start
                    endu = sm < trajs[ku].id_start
                    if endl and not endu:
                        b = 0.5
                        kl, _ = parents[kl]
                        a = a + b * (1 - a)
                    if endu and not endl:
                        b = 0.5
                        _, ku = parents[ku]
                        a = a * b
                    if endl and endu:
                        b = 0.5
                        c = 0.5
                        kl, _ = parents[kl]
                        _, ku = parents[ku]
                        a = a * c + b * (1 - a)
                    ptraj_l = self.trajs[kl]
                    ptraj_u = self.trajs[ku]
                    # Position cursor to lowest common instant
                    sb = max(ptraj_l.id_start, ptraj_u.id_start)
                    # Create interpolated traj
                    for s in range(sb, sm + 1):
                        ts[s] = ptraj_l[s].t
                        state = (1 - a) * ptraj_l[s].state + a * ptraj_u[s].state
                        adjoint = (1 - a) * ptraj_l[s].adjoint + a * ptraj_u[s].adjoint
                        points[s] = state
                        adjoints[s] = adjoint
                        controls[s] = self.mp.control_angle(adjoint, state)
                    sm = sb - 1
            if not using_primal:
                ts = ts[::-1]
                points = points[::-1]
                adjoints = adjoints[::-1]
                controls = controls[::-1]
            traj = AugmentedTraj(ts, points, adjoints, controls, nt - 1, mp.coords,
                                 info=f'opt_m{self.mode}_{self.asp_init:.0f}')
            btrajs[k0] = traj
        res = EFOptRes(status, bests, btrajs, self.mp_primal)
        return res

    def get_trajs(self, primal_only=False, dual_only=False):
        res = []
        trajgroups = []
        if not primal_only:
            trajgroups.append(self.trajs_dual)
        if not dual_only:
            trajgroups.append(self.trajs_primal)
        for i, trajs in enumerate(trajgroups):
            for k, p_traj in trajs.items():
                n = len(p_traj.particles)
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
                for it, pcl in enumerate(p_traj.particles):
                    timestamps[it] = pcl.t + offset
                    points[it, :] = pcl.state
                    adjoints[it, :] = pcl.adjoint
                    controls[it] = self.mp.control_angle(pcl.adjoint, pcl.state)
                    transver[it] = 1 - self.mp.model.v_a * np.linalg.norm(pcl.adjoint) \
                                   + pcl.adjoint @ self.mp.model.wind.value(pcl.t, pcl.state)
                    energy[it] = pcl.cost
                add_info = ''
                if i == 0:
                    try:
                        add_info = '_' + self.trajs_add_info[k]
                    except KeyError:
                        pass
                res.append(
                    AugmentedTraj(timestamps, points, adjoints, controls, last_index=n - 1, coords=self.mp.coords,
                                  label=k,
                                  type=TRAJ_PMP, info=f'ef_{i}{add_info}', transver=transver, energy=energy))

        return res
