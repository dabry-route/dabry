import warnings
from abc import ABC
from typing import Optional, Dict

import numpy as np
import scipy.integrate as scitg
from numpy import ndarray
from tqdm import tqdm

from dabry.misc import directional_timeopt_control, Utils
from dabry.misc import terminal
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


class Site:
    def __init__(self, t_init: float, index_t: int,
                 state_origin: ndarray, costate_origin: ndarray,
                 obstacle_name="", index_t_obs: int = 0):
        self.t_init = t_init
        self.index_t = index_t
        self.state_origin = state_origin.copy()
        self.costate_origin = costate_origin.copy()
        self.traj: Optional[Trajectory] = None
        self.id_check_right = index_t
        self.obstacle_name = obstacle_name
        self.index_t_obs = index_t_obs
        self.closed = False

    def in_obs_at(self, index: int):
        if len(self.obstacle_name) == 0:
            return False
        if index >= self.index_t_obs:
            return True
        return False

    def assign_traj(self, traj: Trajectory):
        self.traj = traj

    def traj_at_index(self, index: int):
        return np.concatenate((self.traj.states[index - self.index_t], self.traj.costates[index - self.index_t]))

    def control_at_index(self, index: int):
        return self.traj.controls[index - self.index_t]


class SolverEF(ABC):
    _ALL_MODES = ['time', 'energy']

    def __init__(self, mp: NavigationProblem,
                 total_duration: float,
                 mode: str = 'time',
                 max_depth: int = 10,
                 n_time: int = 100,
                 n_costate_angle: int = 30,
                 t_init: Optional[float] = None,
                 n_costate_norm: int = 10,
                 costate_norm_bounds: tuple[float] = (0., 1.),
                 target_radius: Optional[float] = None):
        if mode not in self._ALL_MODES:
            raise ValueError('Mode %s is not defined' % mode)
        if mode == 'energy':
            raise ValueError('Mode "energy" is not implemented yet')
        # Problem is assumed to be well conditioned ! (non-dimensionalized)
        self.mp = mp
        # Same for time
        self.total_duration = total_duration
        self.t_init = t_init if t_init is not None else self.mp.model.ff.t_start
        self.target_radius = target_radius if target_radius is not None else \
            0.025 * np.linalg.norm(self.mp.tr - self.mp.bl)
        self._target_radius_sq = self.target_radius ** 2
        self.times = np.linspace(self.t_init, self.t_upper_bound, n_time)
        self.costate_norm_bounds = costate_norm_bounds
        self.n_costate_angle = n_costate_angle
        self.n_costate_norm = n_costate_norm
        self.max_depth = max_depth
        self.success = False
        self.depth = 0
        self.trajs: list[Trajectory] = []
        self.traj_groups: list[tuple[Trajectory]] = []
        self._id_optitraj: int = 0
        self.events = {'target': self._event_target}
        self.obstacles = {}
        classes = {}
        for obs in self.mp.obstacles:
            name = obs.__class__.__name__
            n = classes.get(name)
            if n is None:
                classes[name] = 0
            else:
                classes[name] += 1
            full_name = name + '_' + str(n if n is not None else 0)
            self.events[full_name] = obs.event
            self.obstacles[full_name] = obs
        self._events = list(self.events.values())
        self.obs_active = None
        self.obs_active_trigo = True

        self.dyn_augsys = self.dyn_augsys_cartesian if self.mp.coords == Utils.COORD_CARTESIAN else self.dyn_augsys_gcs

    def setup(self):
        self.trajs = []
        self.traj_groups = []

    @property
    def t_upper_bound(self):
        return self.t_init + self.total_duration

    def dyn_augsys(self, t: float, y: ndarray):
        pass

    def dyn_augsys_cartesian(self, t: float, y: ndarray):
        return self.mp.augsys_dyn_timeopt_cartesian(t, y[:2], y[2:])

    def dyn_augsys_gcs(self, t: float, y: ndarray):
        return self.mp.augsys_dyn_timeopt_gcs(t, y[:2], y[2:])

    def dyn_constr(self, t: float, x: ndarray):
        sign = (2. * self.obs_active_trigo - 1.)
        d = np.array(((0., -sign), (sign, 0.))) @ self.obs_active.d_value(x)
        return directional_timeopt_control(self.mp.model.ff.value(t, x), d, self.mp.srf_max)

    @terminal
    def _event_target(self, _, x):
        return np.sum(np.square(x[:2] - self.mp.x_target)) - self._target_radius_sq

    @terminal
    def _event_quit_obs(self, t, x):
        n = self.obs_active.d_value(x) / np.linalg.norm(self.obs_active.d_value(x))
        ff_val = self.mp.model.ff.value(t, x)
        ff_ortho = ff_val @ n
        return self.mp.srf_max - np.abs(ff_ortho)

    def t_events_to_dict(self, t_events: list[ndarray]) -> Dict[str, ndarray]:
        d = {}
        for i, name in enumerate(self.events.keys()):
            d[name] = t_events[i]
        return d

    @property
    def n_time(self):
        return self.times.shape[0]

    def get_traj_and_id(self, depth: int) -> list[tuple[int, Trajectory]]:
        res = []
        for traj in self.traj_groups[depth]:
            res.append((self.trajs.index(traj), traj))
        return res

    def cost_map(self, nx: int = 100, ny: int = 100) -> ndarray:
        res = np.nan * np.ones((nx, ny))
        spacings = (self.mp.tr - self.mp.bl) / (np.array((nx, ny)) - np.ones(2).astype(np.int32))
        for traj in self.trajs:
            for k, state_aug in enumerate(traj.states):
                position = (state_aug - self.mp.bl) / spacings
                index = tuple(
                    np.clip(np.floor(position).astype(np.int32), np.array((0, 0)), np.array((nx - 1, ny - 1))))
                if np.isnan(res[index]) or res[index] > traj.times[k]:
                    res[index] = traj.times[k]
        return res


class SolverEFBisection(SolverEF):

    def __init__(self, *args, **kwargs):
        super(SolverEFBisection, self).__init__(*args, **kwargs)
        self.layers: list[tuple[float]] = []
        self._tested_layers: list = []
        self._id_group_optitraj: int = 0

    def setup(self):
        super().setup()
        self.layers = []
        self._tested_layers = []

    def step(self):
        angles = np.linspace(0., 2 * np.pi, 2 ** self.depth * self.n_costate_angle + 1)
        if self.depth != 0:
            angles = angles[1::2]
        costate_init = np.stack((np.cos(angles), np.sin(angles)), -1)
        traj_group: list[Trajectory] = []
        for costate in tqdm(costate_init):
            t_eval = self.times
            res = scitg.solve_ivp(self.dyn_augsys, (self.t_init, self.t_upper_bound),
                                  np.array(tuple(self.mp.x_init) + tuple(costate)), t_eval=t_eval,
                                  events=self._events)
            traj = Trajectory.cartesian(t_eval, res.y.transpose()[:, :2], costates=res.y.transpose()[:, 2:],
                                        events=self.t_events_to_dict(res.t_events))
            if traj.events['target'].shape[0] > 0:
                self.success = True
            traj_group.append(traj)
            self.trajs.append(traj)
        self.traj_groups.append(tuple(traj_group))
        self.layers.append(tuple(angles))
        self.depth += 1


class SolverEFResampling(SolverEF):

    def __init__(self, *args, max_dist: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_dist = max_dist if max_dist is not None else self.target_radius
        self._max_dist_sq = self.max_dist ** 2
        self._sites: list[Site] = []
        self._site_groups: list[tuple[Site]] = []
        self._to_shoot_sites: list[Site] = []
        self.solution_sites: list[Site] = []

    def setup(self):
        super().setup()
        self._to_shoot_sites = [Site(self.t_init, 0, self.mp.x_init, np.array((np.cos(theta), np.sin(theta))))
                                for theta in np.linspace(0, 2 * np.pi, self.n_costate_angle, endpoint=False)]
        self._sites = [site for site in self._to_shoot_sites]
        self._site_groups = [tuple(self._sites)]

    def step(self):
        trajs: list[Trajectory] = []
        for site in tqdm(self._to_shoot_sites):
            t_eval = np.linspace(site.t_init, self.t_upper_bound, self.n_time - site.index_t)
            if t_eval.shape[0] <= 1:
                continue
            res = scitg.solve_ivp(self.dyn_augsys, (site.t_init, self.t_upper_bound),
                                  np.array(tuple(site.state_origin) + tuple(site.costate_origin)),
                                  t_eval=t_eval,
                                  events=self._events, dense_output=True, max_step=0.5e-2 * self.total_duration)
            if len(res.t) == 0:
                continue
            traj = Trajectory.cartesian(res.t, res.y.transpose()[:, :2], costates=res.y.transpose()[:, 2:],
                                        events=self.t_events_to_dict(res.t_events))
            if traj.events['target'].shape[0] > 0:
                self.success = True
                self.solution_sites.append(site)
            active_obstacles = [name for name, t_events in traj.events.items()
                                if t_events.shape[0] > 0 and name != 'target']
            if len(active_obstacles) >= 2:
                warnings.warn("Multiple active obstacles", category=RuntimeWarning)
            if len(active_obstacles) >= 1:
                obs_name = active_obstacles[0]
                t_enter_obs = traj.events[obs_name][0]
                self.obs_active = self.obstacles[obs_name]
                site.obstacle_name = obs_name
                site.index_t_obs = site.index_t + res.t.shape[0]
                state_aug_cross = res.sol(t_enter_obs)
                cross = np.cross(self.obs_active.d_value(state_aug_cross[:2]),
                                 self.mp.model.dyn.value(t_enter_obs, state_aug_cross[:2],
                                                         -state_aug_cross[2:] / np.linalg.norm(state_aug_cross[2:])))
                self.obs_active_trigo = cross >= 0.
                res = scitg.solve_ivp(self.dyn_constr, (t_enter_obs, self.t_upper_bound), state_aug_cross[:2],
                                      t_eval=t_eval[t_eval > t_enter_obs], events=[self._event_quit_obs],
                                      max_step=0.5e-2 * self.total_duration)
                if len(res.t) > 0:
                    controls = np.array([self.dyn_constr(t, x) - self.mp.model.ff.value(t, x)
                                         for t, x in zip(res.t, res.y.transpose())])
                    traj_obs = Trajectory.cartesian(res.t, res.y.transpose(), controls=controls)
                    traj = traj + traj_obs
            trajs.append(traj)
            self.trajs.append(traj)
            site.assign_traj(traj)
        self._to_shoot_sites = []
        self.add_new_sites()
        self.depth += 1

    def get_traj_and_id(self, depth: int) -> list[tuple[int, Trajectory]]:
        res = []
        for site in self._site_groups[depth]:
            res.append((self._sites.index(site), site.traj))
        return res

    def add_new_sites(self):
        prev_sites = [site for site in self._sites]
        new_sites: list[Site] = []
        for i_s, site in enumerate(prev_sites):
            if site.closed:
                continue
            site_nb = prev_sites[(i_s + 1) % len(prev_sites)]
            new_id_check_right = site.id_check_right
            new_site: Optional[Site] = None
            for i in range(site.id_check_right, self.n_time - 1):
                try:
                    state_aug = site.traj_at_index(i)
                    state_aug_nb = site_nb.traj_at_index(i)
                    new_id_check_right = i
                    if site.in_obs_at(i):
                        if site_nb.in_obs_at(i):
                            site.closed = True
                            break
                        state_aug[2:] = -site.control_at_index(i) * np.linalg.norm(state_aug_nb[2:])
                    if site_nb.in_obs_at(i):
                        state_aug_nb[2:] = -site_nb.control_at_index(i) * np.linalg.norm(state_aug[2:])
                    if np.sum(np.square((state_aug - state_aug_nb)[:2])) > self._max_dist_sq:
                        new_site = Site(self.times[i], i, 0.5 * (state_aug[:2] + state_aug_nb[:2]),
                                        0.5 * (state_aug[2:] + state_aug_nb[2:]))
                        break
                except IndexError:
                    continue
            site.id_check_right = new_id_check_right
            i_site = self._sites.index(site)
            if new_site is not None:
                self._sites = self._sites[:i_site + 1] + [new_site] + self._sites[i_site + 1:]
                self._to_shoot_sites.append(new_site)
                new_sites.append(new_site)

        self._site_groups.append(tuple(new_sites))
