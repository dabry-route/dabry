import csv
import os
from abc import ABC
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from alphashape import alphashape
from numpy import arcsin as asin, ndarray
from numpy import arctan2 as atan2
from numpy import sin, cos, pi
from shapely.geometry import Point

from dabry.misc import Utils, Chrono, Debug
from dabry.problem import NavigationProblem
from dabry.trajectory import Trajectory

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
        """
        if len(particles) > 0:
            # Check consistency
            assert (particles[0].idt == id_start)
            for i in range(len(particles) - 1):
                assert (particles[i + 1].idt - particles[i].idt == 1)
                assert (particles[i + 1].idf == self.id_traj)
        """
        self.id_start = id_start
        self.particles = particles

    def append(self, p: Particle):
        # assert (p.idt - self.points[-1].idt == 1)
        assert (p.idf == self.id_traj)
        self.particles.append(p)

    def to_traj(self, coords):
        n = len(self.particles)
        times = np.zeros(n)
        states = np.zeros((n, 2))
        adjoints = np.zeros((n, 2))
        for i, pcl in enumerate(self.particles):
            times[i] = pcl.t
            states[i, :] = pcl.state
            adjoints[i, :] = pcl.adjoint
        return Trajectory(times, states, coords, costates=adjoints)

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

    def load_from_trajs(self, directory, filename=None, aero=None):
        if filename is None:
            names = os.listdir(directory)
            for name in names:
                if 'trajectories' in name:
                    filename = name
                    break
        with h5py.File(os.path.join(directory, filename)) as f:
            for k, traj in enumerate(f.values()):
                # Filter extremal fields
                if 'info' in traj.attrs.keys() and 'ef_' in traj.attrs['info']:
                    continue
                duration = traj['ts'][-1] - traj['ts'][0]
                if 'constant_airspeed' in traj.attrs.keys() and aero is not None:
                    asp = traj.attrs['constant_airspeed']
                    energy = duration * aero.power(asp)
                    self.add((duration, energy))
                else:
                    energy = traj['energy'][-1] - traj['energy'][0]
                    self.add((duration, energy))

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
            self.durations = self.durations[:i0 + 1] + [t] + self.durations[i:]
            self.energies = self.energies[:i0 + 1] + [e] + self.energies[i:]


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
        :param mp: NavigationProblem
        """
        self.status = status
        self.bests = bests
        self.trajs = trajs
        self.mp = mp
        try:
            self.min_cost_idx = sorted(bests, key=lambda k: bests[k].cost)[0]
        except IndexError:
            self.min_cost_idx = -1

    @property
    def cost(self):
        return self.bests[self.min_cost_idx].cost

    @property
    def duration(self):
        return self.bests[self.min_cost_idx].t - self.mp.model.ff.t_start

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


class Edge:

    def __init__(self, i1, i2):
        self.i1, self.i2 = (i1, i2) if i1 < i2 else (i2, i1)
        self._i1, self._i2 = (i1, i2) if i1 < i2 else (i2, i1)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self._i1 == other._i1 and self._i2 == other._i2

    def __contains__(self, item):
        return self.i1 == item or self.i2 == item

    def first(self):
        return self.i1

    def second(self):
        return self.i2


class Triangle:

    def __init__(self, a: int, b: int, c: int):
        """
        :param a: Identifier of first vertex
        :param b: Identifier of second vertex
        :param c: Identifier of third vertex
        """
        self.a, self.b, self.c = tuple(sorted((a, b, c)))
        self._a, self._b, self._c = a, b, c

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return False
        return self.a == other.a and self.b == other.b and self.c == other.c

    def __contains__(self, item):
        return item in [self.a, self.b, self.c]


class TriColl:

    def __init__(self, tri_list):
        """
        Collection of triangles handling rapid access to triangle for vertices
        :param tri_list: List of Triangle objects
        """
        self.tris = []
        self.whos_tri = []
        for tri in tri_list:
            self.add(tri)

    def add(self, tri):
        self.tris.append(tri)


class ExtremalField(ABC):

    def __init__(self):
        self.trajs = []
        self.p_inits = {}
        self.parents = {}
        self.active = []
        self.captive = []
        self.deact_reason = {}
        self.deact_buffer = []
        self.edges = []
        self.index = 0

    def _create_id(self):
        i = self.index
        self.index += 1
        self.trajs.append(None)
        return i

    def new_traj(self, pcl: Particle, p_init, i=None):
        """
        Create new initial trajectory
        """
        if i is None:
            i = self._create_id()
        self.trajs[i] = PartialTraj(pcl.idt, [pcl])
        self.p_inits[i] = np.zeros(2)
        self.p_inits[i][:] = p_init
        self.active.append(i)

    def add(self, pcl: Particle):
        """
        Add particle to existing trajectory
        """
        id_traj = pcl.idf
        if id_traj >= len(self.trajs):
            raise Exception(f'Error in add : Trajectory {id_traj} does not exist')
        else:
            self.trajs[id_traj].append(pcl)

    def get_parents(self, i):
        try:
            return self.parents[i]
        except KeyError:
            return None, None

    def get_init_adjoint(self, i):
        if i not in self.parents.keys():
            return self.p_inits[i]
        else:
            p1, p2 = self.parents[i]
            return 0.5 * (self.get_init_adjoint(p1) + self.get_init_adjoint(p2))

    def empty(self):
        return len(self.trajs) == 0

    def get_nb(self, i):
        nb = []
        for e in self.edges:
            if i in e:
                if e[0] == i:
                    nb.append(e[1])
                else:
                    nb.append(e[0])
        return nb

    def is_active(self, i):
        return i in self.active

    def set_captive(self, i):
        self.captive.append(i)

    def deactivate(self, i, reason=''):
        if i in self.active:
            self.active.remove(i)
            self.deact_reason[i] = reason
            self.deact_buffer.append(i)

    def last_active_pcls(self):
        res = []
        for i in self.active:
            res.append(self.trajs[i].get_last())
        return res

    def resample(self, *args):
        pass

    def setup(self, *args):
        pass

    def prune(self):
        pass


class ExtremalField2D(ExtremalField):

    def __init__(self):
        super().__init__()

    def setup(self, N, t_start, x_init):
        for k in range(N):
            theta = 2 * np.pi * k / N
            p = np.array((np.cos(theta), np.sin(theta)))
            """
            if self.mode == self.MODE_ENERGY:
                if self.mp.model.wind.t_end is not None or self.no_transversality:
                    asp = self.asp_init
                else:
                    asp = self.asp_offset + self.mp.aero.asp_mlod(
                        -pd @ self.mp.model.wind.value(t_start, self.mp.x_init))
                pn = self.mp.aero.d_power(asp)
            """
            pcl = Particle(k, 0, t_start, x_init, p, 0.)
            self.new_traj(pcl, p)
            self.edges.append(Edge(k, (k + 1) % N))

    def resample(self, edge):
        i1 = edge.first()
        i2 = edge.second()
        idt = self.trajs[i1].get_last().idt

        pcl1 = self.trajs[i1][idt]
        pcl2 = self.trajs[i2][idt]

        self.edges.remove(Edge(i1, i2))

        i_new = self._create_id()
        state = 0.5 * (pcl1.state + pcl2.state)
        cost = 0.5 * (pcl1.cost + pcl2.cost)
        adjoint = 0.5 * (pcl1.adjoint + pcl2.adjoint)
        pcl_new = Particle(i_new, idt, pcl1.t, state, adjoint, cost)

        self.edges.append(Edge(i1, i_new))
        self.edges.append(Edge(i_new, i2))
        self.trajs[i_new] = PartialTraj(pcl_new.idt, [pcl_new])
        self.parents[i_new] = (i1, i2)
        self.active.append(i_new)

        return pcl_new

    def prune(self):
        """
        Prune all links to inactive particles
        """
        edges = []
        for e in self.edges:
            if e.first() in self.deact_buffer or e.second() in self.deact_buffer:
                continue
            edges.append(e)
        self.edges = edges
        self.deact_buffer.clear()


class ExtremalField2DE(ExtremalField):

    def __init__(self):
        super().__init__()
        self.tris = []
        self.my_tris = []

    def _create_id(self):
        self.my_tris.append([])
        return super()._create_id()

    def add_tri(self, tri):
        self.tris.append(tri)
        self.my_tris[tri.a].append(tri)
        self.my_tris[tri.b].append(tri)
        self.my_tris[tri.c].append(tri)

    def rm_tri(self, tri):
        self.tris.remove(tri)
        self.my_tris[tri.a].remove(tri)
        self.my_tris[tri.b].remove(tri)
        self.my_tris[tri.c].remove(tri)

    def are_linked(self, i1, i2):
        for tri in self.my_tris[i1]:
            if i2 in tri:
                return True
        return False

    def common(self, i1, i2):
        s1 = set()
        s2 = set()
        for tri in self.my_tris[i1]:
            if i2 in tri:
                continue
            s1.add(tri.a)
            s1.add(tri.b)
            s1.add(tri.c)
        for tri in self.my_tris[i2]:
            if i1 in tri:
                continue
            s2.add(tri.a)
            s2.add(tri.b)
            s2.add(tri.c)
        s1.remove(i1)
        s2.remove(i2)
        res = s1.intersection(s2)
        if len(res) >= 2:
            raise Exception('Intersection containing more than one point')
        elif len(res) == 0:
            return None
        else:
            return res.pop()

    def setup(self, Nph: int, Na: int, t_start: float, x_init: ndarray, pn_bounds: tuple[float, float]):
        """
        :param Nph: Discretization number in phase (angles)
        :param Na: Discretization number in amplitude
        :param t_start: Starting date for particles in seconds (POSIX).
        :param x_init: (2,) array to initialize particles
        :param pn_bounds: Minimum and maximum value for adjoint norm
        There are na rows of nph points distributed in circles.
        ...-o----o----o----o-...
             \  / \  / \  /
        ...---o----o----o---...
             /  \ /  \ /  \
        ...-o----o----o----o-...
        Relationships go like 0 <-> 1 <-> 2 <-> ... <-> nph-1 and nph-1 <-> 0 within a circle.
        """
        grid = np.zeros((Na, Nph), dtype=int)
        for ka in range(Na):
            for kph in range(Nph):
                k = self._create_id()
                grid[ka, kph] = k
                offset = 0 if ka % 2 == 0 else 0.5
                theta = 2 * np.pi * (kph + offset) / Nph
                pn = pn_bounds[0] + ka / (Na - 1) * (pn_bounds[1] - pn_bounds[0])
                p = pn * np.array((np.cos(theta), np.sin(theta)))
                """
                if self.mp.model.wind.t_end is not None or self.no_transversality:
                    asp = self.asp_init
                else:
                    asp = self.asp_offset + self.mp.aero.asp_mlod(
                        -pd @ self.mp.model.wind.value(t_start, self.mp.x_init))
                pn = self.mp.aero.d_power(asp)
                """
                pcl = Particle(k, 0, t_start, x_init, p, 0.)
                self.new_traj(pcl, p, i=k)

        # Create connections
        for ka in range(Na):
            for kph in range(Nph):
                i = grid[ka, kph]
                # Inner circle neighbours
                ir = grid[ka, (kph + 1) % Nph]
                # self.edges.append(Edge(i, ir))
                if ka != Na - 1:
                    # Upper neighbours. A member at position i in an even row is in
                    # relation with the i-1 and i positions in the upper row.
                    # In an odd row, it is in relation with i and i+1
                    if ka % 2 == 0:
                        iul = grid[ka + 1, (kph - 1) % Nph]
                        iur = grid[ka + 1, kph]
                    else:
                        iul = grid[ka + 1, kph]
                        iur = grid[ka + 1, (kph + 1) % Nph]
                    # self.edges.append(Edge(i, iul))
                    # self.edges.append(Edge(i, iur))
                    t1 = Triangle(i, ir, iur)
                    t2 = Triangle(i, iul, iur)
                    self.add_tri(t1)
                    self.add_tri(t2)

    def resample(self, tri):
        i_a, i_b, i_c = tri.a, tri.b, tri.c

        pcla = self.trajs[i_a].get_last()
        pclb = self.trajs[i_b].get_last()
        pclc = self.trajs[i_c].get_last()

        # Resample by under-scaling triangle
        # Naming conventions
        #         a
        #         o
        # gamma /   \ beta
        #      /     \
        #     o-------o
        #  b    alpha    c

        # Some edges may already be split
        split_alpha = not self.are_linked(i_b, i_c)
        split_beta = not self.are_linked(i_a, i_c)
        split_gamma = not self.are_linked(i_a, i_b)

        to_create = []
        if split_alpha:
            i_alpha = self.common(i_b, i_c)
        else:
            i_alpha = self._create_id()
            to_create.append((i_alpha, pclb, pclc))
        if split_beta:
            i_beta = self.common(i_a, i_c)
        else:
            i_beta = self._create_id()
            to_create.append((i_beta, pcla, pclc))
        if split_gamma:
            i_gamma = self.common(i_a, i_b)
        else:
            i_gamma = self._create_id()
            to_create.append((i_gamma, pcla, pclb))

        s = set()
        s.add(i_alpha)
        s.add(i_beta)
        s.add(i_gamma)
        if len(s) <= 2:
            raise Exception('Flat triangle')

        new_pcls = []
        for i_new, pcl1, pcl2 in to_create:
            state = 0.5 * (pcl1.state + pcl2.state)
            cost = 0.5 * (pcl1.cost + pcl2.cost)
            adjoint = 0.5 * (pcl1.adjoint + pcl2.adjoint)
            pcl_new = Particle(i_new, pcl1.idt, pcl1.t, state, adjoint, cost)
            self.trajs[i_new] = PartialTraj(pcl_new.idt, [pcl_new])
            self.parents[i_new] = (pcl1.idf, pcl2.idf)
            self.active.append(i_new)
            new_pcls.append(pcl_new)

        self.add_tri(Triangle(i_a, i_beta, i_gamma))
        self.add_tri(Triangle(i_b, i_alpha, i_gamma))
        self.add_tri(Triangle(i_c, i_alpha, i_beta))
        self.add_tri(Triangle(i_alpha, i_beta, i_gamma))

        self.rm_tri(tri)

        return new_pcls

    def prune(self):
        to_delete = []
        for tri in self.tris:
            if tri.a in self.deact_buffer or tri.b in self.deact_buffer or tri.c in self.deact_buffer:
                to_delete.append(tri)

        for tri in to_delete:
            self.rm_tri(tri)


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
                 mp: NavigationProblem,
                 t_init: Optional[float] = None,
                 max_time=None,
                 mode=0,
                 N_disc_init=20,
                 rel_nb_ceil=0.02,
                 dt=None,
                 max_steps=None,
                 collbuf_shape=None,
                 cost_ceil=None,
                 rel_cost_ceil=None,
                 asp_offset=0.,
                 asp_init=None,
                 no_coll_filtering=False,
                 quick_solve=False,
                 quick_offset=0,
                 pareto=None,
                 max_active_ext=None,
                 no_transversality=False,
                 v_bounds=None,
                 N_pn_init=5,
                 alpha_factor=1.):
        self.mp_primal = mp
        self.mp_dual = mp.dualize()

        self.t_init = t_init

        self.mode = mode

        if self.mode == 1:
            if v_bounds is None:
                self.v_min, self.v_max = mp.aero.v_min, mp.aero.v_max
            else:
                self.v_min, self.v_max = v_bounds

        # Solve time arguments
        if sum([max_time is None, dt is None, max_steps is None]) >= 2:
            if max_steps is not None:
                # Trying automatic parameters
                dt = 1.5 * mp.l_ref / mp.aero.v_min / max_steps
                print(f'Automatic dt ({dt:.5g})')
            else:
                raise Exception('Please provide at least two arguments among "max_time", "max_steps" and "dt"')

        if max_time is None:
            max_time = dt * max_steps

        self.N_disc_init = N_disc_init
        self.N_pn_init = N_pn_init
        # Neighbouring distance ceil
        self.abs_nb_ceil = rel_nb_ceil * mp.geod_l
        if dt is not None:
            self.dt = -dt
        else:
            if mode == 0:
                self.dt = -.005 * mp.l_ref / mp.srf_max
            else:
                self.dt = -.005 * mp.l_ref / mp.aero.v_minp

        if max_steps is None:
            max_steps = int(abs(max_time / dt))

        # This group shall be reinitialized before new resolution
        self.ef_primal = ExtremalField2D() if self.mode == 0 else ExtremalField2DE()
        self.ef_dual = ExtremalField2D() if self.mode == 0 else ExtremalField2DE()
        self.ef = self.ef_primal
        self.lsets = []
        self.n_points = 0
        self.no_transversality = no_transversality

        self.max_active_ext = max_active_ext

        self.trajs_control = {}
        self.trajs_add_info = {}
        # Default mode is dual problem
        self.mode_primal = False
        self.mp = self.mp_dual
        self.max_extremals = 10000
        self.max_time = max_time
        if cost_ceil is None and mode == 1:
            cost_ceil = self.mp.aero.power(self.v_max) * self.mp.l_ref / self.v_max
            print(f'Automatic cost_ceil ({cost_ceil:.5g})')
        self.cost_ceil = cost_ceil
        self.rel_cost_ceil = 0.05 if rel_cost_ceil is None else rel_cost_ceil
        self.cost_thres = None if cost_ceil is None else rel_cost_ceil * cost_ceil
        self.asp_offset = asp_offset
        self.asp_init = asp_init if asp_init is not None else self.mp.srf_max

        self.it = 0
        self.it_first_reach = -1

        self.collbuf_shape = (50, 50) if collbuf_shape is None else collbuf_shape
        self.collbuf = CollisionBuffer()

        self.reach_duration = None
        self.reach_cost = None

        self._index_p = 0
        self._index_d = 0

        self.max_steps = max_steps if max_steps is not None else min(int(max_time / abs(self.dt)), 1000)

        self.N_filling_steps = 1

        # Base collision filtering scheme only for steady problems
        self.no_coll_filtering = no_coll_filtering or self.mp.model.ff.t_end is not None
        self.quick_solve = quick_solve if mode == 1 else True
        self.quick_offset = quick_offset
        self._qcounter = 0
        self.quick_traj_idx = -1
        self.quick_exit = False
        self.best_distance = self.mp.geod_l

        self.any_out_obs = False

        self.manual_integration = True

        self.step_algo = 0

        self.pareto = pareto

        self.trimming_rate = 10

        # Parameter for alpha shapes in the trimming procedure
        self.alpha_param = 0.05 * alpha_factor
        self.alpha_value = 1 / (self.alpha_param * self.mp.l_ref)

        # For debug
        self.reg = Register()

    def set_primal(self, b):
        self.mode_primal = b
        if b:
            self.mp = self.mp_primal
            self.ef = self.ef_primal
            if self.dt < 0:
                self.dt *= -1
        else:
            self.mp = self.mp_dual
            self.ef = self.ef_dual
            if self.dt > 0:
                self.dt *= -1

    def setup(self, time_offset=0.):
        """
        Push all possible angles to active list from init point
        :return: None
        """
        if self.mode_primal:
            self.ef_primal = ExtremalField2D() if self.mode == 0 else ExtremalField2DE()
            self.ef = self.ef_primal
        else:
            self.ef_dual = ExtremalField2D() if self.mode == 0 else ExtremalField2DE()
            self.ef = self.ef_dual
        self.lsets = []
        self.it = 0
        self.n_points = 0
        self.any_out_obs = True
        if self.t_init is not None:
            t_start = self.t_init + time_offset
        else:
            t_start = self.mp_primal.model.ff.t_start + time_offset
        if not self.no_coll_filtering:
            self.collbuf.init(*self.collbuf_shape, self.mp.bl, self.mp.tr)

        if self.mode == 0:
            self.ef.setup(self.N_disc_init, t_start, self.mp.x_init)
        if self.mode == 1:
            p_min = self.mp.aero.d_power(self.mp.aero.v_min)
            p_max = self.mp.aero.d_power(self.mp.aero.v_max)
            self.ef.setup(self.N_disc_init, self.N_pn_init, t_start, self.mp.x_init, (p_min, p_max))

        for pcl in self.ef.last_active_pcls():
            if not self.no_coll_filtering:
                self.collbuf.add_particle(pcl)

    def step_global(self):

        self.any_out_obs = False
        self.best_distance = self.mp.geod_l

        # Evolve active particles
        copy_active = []
        for i in self.ef.active:
            copy_active.append(i)

        up_qcounter = False

        for idf in copy_active:
            pcl = self.ef.trajs[idf].get_last()

            pcl_new, status = self.step_single(pcl)

            if pcl_new.i_obs == -1:
                self.any_out_obs = True
            elif pcl.i_obs == -1:
                self.ef.set_captive(idf)

            dist = self.mp.distance(pcl_new.state, self.mp.x_target)
            if dist < self.best_distance:
                self.best_distance = dist
            if dist < self.abs_nb_ceil:
                if self.it_first_reach == -1:
                    self.it_first_reach = self.it
                    self.quick_traj_idx = idf
                if self.quick_solve:
                    if self._qcounter == self.quick_offset:
                        self.quick_exit = True
                    else:
                        if not up_qcounter:
                            self._qcounter += 1
                            up_qcounter = True
            if not status:
                self.ef.deactivate(idf, 'domain')
            elif abs(pcl_new.rot) > 2 * np.pi:
                self.ef.deactivate(idf, 'roundabout')
            elif self.cost_ceil is not None and pcl_new.cost > self.cost_ceil:
                self.ef.deactivate(idf, 'cost')
            elif self.pareto is not None and \
                    self.pareto.dominated((pcl_new.t - self.mp_primal.model.ff.t_start, pcl_new.cost)):
                self.ef.deactivate(idf, 'pareto')
            else:
                self.ef.add(pcl_new)
                self.n_points += 1

        # Collision filtering if needed
        if not self.no_coll_filtering:
            self.collision_filter()

            for pcl in self.ef.last_active_pcls():
                if pcl.i_obs >= 0:
                    # Points in obstacles do not contribute to collision buffer
                    continue
                self.collbuf.add_particle(pcl)

        # Resampling of the front
        self.resample()

        # Trimming procedure
        if self.no_coll_filtering:
            if self.it > 0 and self.it % self.trimming_rate == 0:
                self.trim()

        self.it += 1

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
        v_a = None
        if pcl.i_obs == -1:
            # For the moment, simple Euler scheme
            u = self.mp.control_angle(p, x)
            su = self.mp.srf_max * np.array((np.cos(u), np.sin(u)))
            kw = {}
            if self.mode == self.MODE_ENERGY:
                kw['v_a'] = v_a = np.min(
                    (self.mp.aero.v_max, np.max((self.mp.aero.v_min, self.mp.aero.asp_opti(p)))))
            dyn_x = self.mp.model.dyn.value(t, x, su)
            A = -self.mp.model.dyn.d_value__d_state(t, x, su, **kw).transpose()
            dyn_p = A.dot(p) - self.mp.penalty.d_value(t, x)
            x += self.dt * dyn_x
            p += self.dt * dyn_p
            t += self.dt
            # Transversality condition for debug purposes
            # a = 1 - self.mp.model.v_a * np.linalg.norm(p) + p @ self.mp.model.wind.value(t, x)
            # For compatibility with other case
            status = True
        else:
            # Obstacle mode. Follow the obstacle boundary as fast as possible
            obs_grad = self.mp.obstacles[pcl.i_obs].d_value(x)

            obs_tgt = (-1. if not pcl.ccw else 1.) * np.array(((0, -1), (1, 0))) @ obs_grad

            v_a = self.mp.srf_max

            arg_ot = atan2(*obs_tgt[::-1])
            w = self.mp.model.ff.value(t, x)
            arg_w = atan2(*w[::-1])
            norm_w = np.linalg.norm(w)
            r = norm_w / v_a * sin(arg_w - arg_ot)
            if np.abs(r) >= 1.:
                # Impossible to follow obstacle
                status = False
            else:
                def gs_f(uu, va, w, reference):
                    # Ground speed projected to reference
                    return (np.array((cos(uu), sin(uu))) * va + w) @ reference

                # Select heading that maximizes ground speed along obstacle
                u = max([arg_ot - asin(r), arg_ot + asin(r) - pi], key=lambda uu: gs_f(uu, v_a, w, obs_tgt))
                if self.mp.coords == Utils.COORD_GCS:
                    u = np.pi / 2. - u
                su = self.mp.srf_max * np.array((np.cos(u), np.sin(u)))
                dyn_x = self.mp.model.dyn.value(t, x, su)
                x_prev = np.zeros(x.shape)
                x_prev[:] = x
                x += self.dt * dyn_x
                p[:] = - 1. * np.array((np.cos(u), np.sin(u)))
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
                    if self.mp.coords == Utils.COORD_GCS:
                        u = np.pi / 2 - u
                    s = np.array((np.cos(u), np.sin(u)))
                    obs_grad = self.mp.obstacles[i_obs_new].d_value(x)
                    obs_tgt = np.array(((0, -1), (1, 0))) @ obs_grad
                    ccw = s @ obs_tgt > 0.
                    rot = 0.
        if self.mode == self.MODE_TIME:
            power = self.mp.aero.power(self.mp.srf_max)
        else:
            power = self.mp.aero.power(v_a)
        cost = power * abs(self.dt) + pcl.cost
        new_pcl = Particle(pcl.idf, pcl.idt + 1, t, x, p, cost, i_obs_new, ccw, rot)
        return new_pcl, status

    def resample(self):
        # First prune all links
        self.ef.prune()

        if self.mode == 0:
            copy_edges = []
            for e in self.ef.edges:
                copy_edges.append(e)

            for e in copy_edges:
                i1, i2 = e.first(), e.second()
                pcl1 = self.ef.trajs[i1].get_last()
                pcl2 = self.ef.trajs[i2].get_last()
                # Only resample if precision is lost
                if self.mp.distance(pcl1.state, pcl2.state) <= self.abs_nb_ceil:
                    continue
                # Only resample when at least one point lies outside of an obstacle
                if pcl1.i_obs >= 0 and pcl2.i_obs >= 0:
                    continue

                pcl_new = self.ef.resample(e)

                if not self.no_coll_filtering:
                    self.collbuf.add_particle(pcl_new)
        else:
            copy_tris = []
            for tri in self.ef.tris:
                copy_tris.append(tri)

            for tri in copy_tris:
                i_a, i_b, i_c = tri.a, tri.b, tri.c
                pcla = self.ef.trajs[i_a].get_last()
                pclb = self.ef.trajs[i_b].get_last()
                pclc = self.ef.trajs[i_c].get_last()
                c1 = self.mp.distance(pcla.state, pclb.state) > self.abs_nb_ceil
                c2 = self.mp.distance(pclb.state, pclc.state) > self.abs_nb_ceil
                c3 = self.mp.distance(pcla.state, pclc.state) > self.abs_nb_ceil
                c4 = abs(pcla.cost - pclb.cost) > self.cost_thres
                c5 = abs(pclb.cost - pclc.cost) > self.cost_thres
                c6 = abs(pcla.cost - pclc.cost) > self.cost_thres

                # Only resample when at least one point lies outside of an obstacle
                if sum((pcla.i_obs == -1, pclb.i_obs == -1, pclb.i_obs == -1)) == 0:
                    continue

                resamp_cond = c1 or c2 or c3 or c4 or c5 or c6
                if resamp_cond:
                    self.ef.resample(tri)

    def collision_filter(self):
        for pcl in self.ef.last_active_pcls():
            idf = pcl.idf
            # Deactivate new active points which fall inside front
            if pcl.i_obs >= 0:
                # Skip active points inside obstacles
                continue
            it = pcl.idt
            pcl_prev = self.ef.trajs[idf][pcl.idt - 1]
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
                    pcl1 = self.ef.trajs[k][itt]
                    pcl2 = self.ef.trajs[k][itt + 1]
                    points = x_it, x_itp1, pcl1.state, pcl2.state
                    if Utils.has_intersec(*points):
                        _, alpha1, alpha2 = Utils.intersection(*points)
                        cost1 = (1 - alpha1) * pcl_prev.cost + alpha1 * pcl.cost
                        cost2 = (1 - alpha2) * pcl1.cost + alpha2 * pcl2.cost
                        if cost2 < cost1:
                            self.ef.deactivate(idf, 'collbuf')
                            stop = True
                if stop:
                    break
                p1, p2 = self.ef.get_parents(k)
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
                        pcl1 = self.ef.trajs[k][itt]
                        pcl2 = self.ef.trajs[p][itt]
                        points = x_it, x_itp1, pcl1.state, pcl2.state
                        if Utils.has_intersec(*points):
                            _, alpha1, alpha2 = Utils.intersection(*points)
                            cost1 = (1 - alpha1) * pcl_prev.cost + alpha1 * pcl.cost
                            cost2 = (1 - alpha2) * pcl1.cost + alpha2 * pcl2.cost
                            if cost2 < cost1:
                                self.ef.deactivate(idf, 'collbuf_par')
                                stop = True
                if stop:
                    break

    def trim(self):
        if self.mode == 0:
            pcls = []
            idx = set()
            for pcl in self.ef.last_active_pcls():
                idx.add(pcl.idf)
            for idf in idx:
                traj = self.ef.trajs[idf]
                for pcl in traj.particles[-20:]:
                    pcls.append(pcl)

            # In case of GCS points (lon, lat), multiply by Earth radius
            # to build alpha shape in plate-carree projection space
            factor = Utils.EARTH_RADIUS if self.mp.coords == Utils.COORD_GCS else 1.
            points = [factor * pcl.state for pcl in pcls]
            hull = alphashape(points, self.alpha_value)
            for pcl in self.ef.last_active_pcls():
                if hull.boundary.distance(Point(factor * pcl.state)) > 2 * self.abs_nb_ceil:
                    self.ef.deactivate(pcl.idf, reason='Hull')
        elif self.mode == 1:
            pcls = []
            for traj in self.ef.trajs:
                for pcl in traj.particles[-20:]:
                    pcls.append(pcl)

            def normalize(x1, x2, c):
                xx1 = (x1 - self.mp.bl[0]) / (self.mp.tr[0] - self.mp.bl[0])
                xx2 = (x2 - self.mp.bl[1]) / (self.mp.tr[1] - self.mp.bl[1])
                cc = c / self.cost_ceil
                return xx1, xx2, cc

            n = len(pcls)
            points = np.zeros((n, 3))
            for i, pcl in enumerate(pcls):
                points[i, :] = normalize(pcl.state[0], pcl.state[1], pcl.cost)
            hull = alphashape(points, self.alpha_param)
            for pcl in self.ef.last_active_pcls():
                z = np.array((normalize(*(tuple(pcl.state) + (pcl.cost,))),))
                dist = hull.nearest.on_surface(z)[1][0]
                if dist > 0.05:
                    self.ef.deactivate(pcl.idf, reason='Hull')
        # Apply deactivation to the graph of relations
        self.ef.prune()

    def exit_cond(self):
        return len(self.ef.active) == 0 or \
               self.step_algo >= self.max_steps or \
               len(self.ef.trajs) >= self.max_extremals or \
               self.quick_exit or \
               (not self.any_out_obs) or \
               (self.max_active_ext is not None and len(self.ef.active) > self.max_active_ext)

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
                    f'Extremals : {len(self.ef.trajs):>6}, Active : {len(self.ef.active):>4}, '
                    f'Dist : {self.best_distance / self.mp.geod_l * 100:>3.0f} ', end='', flush=True)
        if self.step_algo == self.max_steps:
            msg = f'Stopped on iteration limit {self.max_steps}'
        elif len(self.ef.trajs) == self.max_extremals:
            msg = f'Stopped on extremals limit {self.max_extremals}'
        elif self.quick_solve and self.quick_traj_idx != -1:
            msg = 'Stopped quick solve'
            self.reach_duration = self.step_algo * self.dt
        elif self.max_active_ext is not None and len(self.ef.active) > self.max_active_ext:
            msg = 'Stopped on maximum of active extremal trajs'
        else:
            msg = f'Stopped empty active list'
        if verbose == 2:
            print(msg)

    def solve(self, verbose=2, backward=False):
        """
        :param verbose: Verbosity level. 2 is common, 1 for multithread, 0 for none.
        :param backward: To run forward and backward extremal computation
        :return: Minimum time to reach destination, corresponding global time index,
        corresponding initial adjoint state
        """

        hello = f'{type(self.mp_primal).__name__}'
        hello += f' | {"TIMEOPT" if self.mode == SolverEF.MODE_TIME else "ENEROPT"}'
        if self.mode == SolverEF.MODE_TIME:
            hello += f' | {self.mp_primal.srf_max:.2f} m/s'
        hello += f' | {self.mp_primal.geod_l:.2e} m'
        nowind_time = self.mp_primal.geod_l / self.mp_primal.srf_max
        hello += f' | scale {Utils.time_fmt(nowind_time)}'
        ortho_time = self.mp_primal.orthodromic()
        hello += f' | orthodromic {Utils.time_fmt(ortho_time) if ortho_time >= 0 else "DNR"}'
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
            self.quick_solve = True
            self.propagate()
            # Now than forward pass is completed, run the backward pass with correct start time
            self.set_primal(False)
            self.quick_solve = False
            self.propagate(time_offset=self.reach_duration, verbose=verbose)
            res = self.build_opti_traj()

        chrono.stop()

        if res.status:
            goodbye = f'Target reached in {Utils.time_fmt(res.duration)}'
            goodbye += f' | {int(100 * (res.duration / nowind_time - 1)):+d}% no wind'
            if ortho_time >= 0:
                goodbye += f' | {int(100 * (res.duration / ortho_time - 1)):+d}% orthodromic'
            goodbye += f' | cpu time {chrono}'
        else:
            goodbye = f'No solution found in time < {Utils.time_fmt(self.max_steps * abs(self.dt))}'
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

    def build_opti_traj(self, dual=False):
        """
        Extract optimal trajectories from completed extremal front.
        May extract several trajectories in energy-optimal mode.
        :param dual: True if problem was solved backward in time.
        :return: EFOptRes
        """
        if not dual:
            self.set_primal(True)
            using_primal = True
        else:
            self.set_primal(False)
            using_primal = False

        bests = {}
        btrajs = {}
        closest = None
        dist = None
        idt_min, idt_max = self.it_first_reach, self.it_first_reach
        if self.quick_exit:
            idt_max += self.quick_offset
            if self.quick_offset == 0:
                bests[self.quick_traj_idx] = self.ef.trajs[self.quick_traj_idx][self.it_first_reach]
        else:
            for k, ptraj in enumerate(self.ef.trajs):
                for idt in range(idt_min, idt_max + 1):
                    try:
                        pcl = ptraj[idt]
                    except IndexError:
                        continue
                    cur_dist = self.mp.distance(pcl.state, self.mp.x_target)
                    if dist is None or cur_dist < dist:
                        dist = cur_dist
                        closest = pcl
                    if cur_dist < self.abs_nb_ceil:
                        bests[k] = pcl
                        break

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
                for k2, b2 in bests.items():
                    if k2 <= k1:
                        continue
                    if k2 in dominated:
                        continue
                    if b2.cost < b1.cost and b2.t < b1.t:
                        dominated.append(k1)
                        break
                    if b1.cost < b2.cost and b1.t < b2.t:
                        dominated.append(k2)
            for k in dominated:
                del bests[k]

        # Build optimal trajectories
        for pcl in bests.values():
            traj = self.traj_rollback(pcl, using_primal)
            btrajs[pcl.idf] = traj
        res = EFOptRes(status, bests, btrajs, self.mp_primal)
        return res

    def traj_rollback(self, pcl: Particle, using_primal=True):
        """
        Builds a trajectory backward from front particle pcl.
        The procedure fills the trajectory with the partial trajectory
        ending at pcl, and further fills backward in time by interpolation
        between parent trajectories.
        :param pcl: Front particle ending trajectory
        :param using_primal: True if problem is forward in time, False else
        :return: AugmentedTraj
        """
        nt = pcl.idt + 1
        k0 = pcl.idf
        p_traj = self.ef.trajs[k0]
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

        kl, ku = self.ef.get_parents(k0)
        if kl is not None:
            a = 0.5
            while sm >= 0:
                endl = sm < self.ef.trajs[kl].id_start
                endu = sm < self.ef.trajs[ku].id_start
                if endl and not endu:
                    b = 0.5
                    kl, _ = self.ef.get_parents(kl)
                    a = a + b * (1 - a)
                if endu and not endl:
                    b = 0.5
                    _, ku = self.ef.get_parents(ku)
                    a = a * b
                if endl and endu:
                    b = 0.5
                    c = 0.5
                    kl, _ = self.ef.get_parents(kl)
                    _, ku = self.ef.get_parents(ku)
                    a = a * c + b * (1 - a)
                ptraj_l = self.ef.trajs[kl]
                ptraj_u = self.ef.trajs[ku]
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
        asp_init = self.asp_init if self.mode == 0 else self.mp.aero.asp_opti(np.linalg.norm(adjoints[0]))
        return AugmentedTraj(ts, points, adjoints, controls, nt - 1, self.mp.coords,
                             info=f'opt_m{int(self.mode)}_{asp_init:.5g}',
                             constant_asp=self.mp.srf_max if self.mode == 0 else None)

    def get_trajs(self, primal_only=False, dual_only=False):
        res = []
        trajgroups = []
        if not primal_only:
            trajgroups.append(self.ef_dual.trajs)
        if not dual_only:
            trajgroups.append(self.ef_primal.trajs)
        for i, trajs in enumerate(trajgroups):
            for k, p_traj in enumerate(trajs):
                n = len(p_traj.particles)
                timestamps = np.zeros(n)
                points = np.zeros((n, 2))
                adjoints = np.zeros((n, 2))
                controls = np.zeros(n)
                transver = np.zeros(n)
                energy = np.zeros(n)
                offset = 0.
                if self.mp_primal.model.ff.t_end is None and i == 0:
                    # If wind is steady, offset timestamps to have a <self.reach_time>-long window
                    # For dual trajectories
                    offset = 0.  # self.reach_time  # - self.max_time
                for it, pcl in enumerate(p_traj.particles):
                    timestamps[it] = pcl.t + offset
                    points[it, :] = pcl.state
                    adjoints[it, :] = pcl.adjoint
                    controls[it] = self.mp.control_angle(pcl.adjoint, pcl.state)
                    transver[it] = 1 - self.mp.srf_max * np.linalg.norm(pcl.adjoint) \
                                   + pcl.adjoint @ self.mp.model.ff.value(pcl.t, pcl.state)
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
                                  type=Utils.TRAJ_PMP, info=f'ef_{i}{add_info}', transver=transver, energy=energy))

        return res

    def get_control_map(self, nx, ny):
        if len(self.ef_dual.trajs) == 0:
            raise Exception('Dual problem not solved')
        grid = np.dstack(np.meshgrid(np.linspace(self.mp.bl[1], self.mp.tr[1], ny),
                                     np.linspace(self.mp.bl[0], self.mp.tr[0], nx))[::-1])

        values = np.zeros((nx - 1, ny - 1))
        costs = np.infty * np.ones((nx - 1, ny - 1))

        x0 = self.mp.bl[0]
        y0 = self.mp.bl[1]
        x1 = self.mp.tr[0]
        y1 = self.mp.tr[1]

        def index(x):
            xx = (x[0] - x0) / (x1 - x0)
            yy = (x[1] - y0) / (y1 - y0)
            return int((nx - 1) * xx), int((ny - 1) * yy)

        for traj in self.ef_dual.trajs.values():
            for pcl in traj.particles:
                i, j = index(pcl.state)
                if pcl.cost < costs[i, j]:
                    costs[i, j] = pcl.cost
                    values[i, j] = self.mp.control_angle(pcl.adjoint, pcl.state)

        return MapFB(grid, values)


class Debugger:

    def __init__(self, solver_ef: SolverEF):
        self.solver = solver_ef
        self.reg = Register()

    def show_points(self):
        ax = plt.figure().add_subplot(projection='3d')
        n = 0
        for traj in self.solver.ef.trajs.values():
            n += len(traj.particles)
        points = np.zeros((n, 3))
        i = 0
        for traj in self.solver.ef.trajs.values():
            for pcl in traj.particles:
                points[i, :] = pcl.state[0], pcl.state[1], pcl.cost
                i += 1

        ax.scatter(points[:10, 0], points[:10, 1], points[:10, 2])
        plt.show()

    def show_points2d(self):
        ax = plt.figure().add_subplot()
        ax.axis('equal')
        n = len(self.solver.ef.trajs)
        points = np.zeros((n, 3))
        labels = np.zeros(n, dtype=int)
        for i, key in enumerate(self.solver.ef.trajs.keys()):
            traj = self.solver.ef.trajs[key]
            pcl = traj.get_last()
            points[i, :] = pcl.state[0], pcl.state[1], pcl.cost
            labels[i] = key

        ax.scatter(points[:, 0], points[:, 1])
        for i in range(n):
            ax.annotate(labels[i], (points[i, 0], points[i, 1]))

        k = 0
        colors = ['red', 'blue', 'yellow', 'purple', 'green', 'orange', 'cyan']
        for tri in self.solver.ef.tris:
            a = self.solver.ef.trajs[tri.a].get_last().state
            b = self.solver.ef.trajs[tri.b].get_last().state
            c = self.solver.ef.trajs[tri.c].get_last().state
            t1 = plt.Polygon((a, b, c), color=colors[(k % len(colors))], alpha=0.5)
            plt.gca().add_patch(t1)
            ax.annotate(self.reg.name_tri(tri), 1 / 3 * (a + b + c))
            k += 1

        plt.show()


class Register:

    def __init__(self):
        self.db = Debug()
        self.names = self.db.names

    def name(self, i: int):
        return self.names[i % len(self.names)]

    def name_tri(self, tri: Triangle):
        i = ((tri.a + 17) * (tri.b + 203)) % len(self.names)
        i = i * (tri.c + 47)
        return self.name(i)
