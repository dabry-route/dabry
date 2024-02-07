import warnings
from abc import ABC
from typing import Optional, Dict, Iterable

import numpy as np
import scipy.integrate as scitg
from numpy import ndarray
from tqdm import tqdm

from dabry.misc import directional_timeopt_control, Utils, triangle_mask_and_cost, non_terminal, to_alpha, \
    is_possible_direction
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

    def __init__(self, t_init: float, index_t_init: int,
                 state_origin: ndarray, costate_origin: ndarray, cost_origin: float, n_time: int,
                 obstacle_name="", index_t_obs: int = 0, name="", init_next_nb=None):
        self.t_init = t_init
        self.index_t_init = index_t_init
        if np.any(np.isnan(costate_origin)):
            raise ValueError('Costate initialization contains NaN')
        self.traj = Trajectory.cartesian(np.array((t_init,)),
                                         state_origin.reshape((1, 2)),
                                         costates=costate_origin.reshape((1, 2)),
                                         cost=np.array((cost_origin,)))
        self.traj_full: Optional[Trajectory] = None
        self.index_t_check_next = index_t_init
        self.obstacle_name = obstacle_name
        self.index_t_obs = index_t_obs
        self.closure_reason = None
        self.next_nb: list[Optional[Site]] = [None] * n_time
        self.name = name
        if init_next_nb is not None:
            self.init_next_nb(init_next_nb)

    @property
    def closed(self):
        return self.closure_reason is not None

    def close(self, reason: str):
        self.closure_reason = reason

    @property
    def has_neighbours(self):
        return any([next_nb is not None for next_nb in self.next_nb])

    @property
    def index_t(self):
        return self.index_t_init + (0 if self.traj is None else len(self.traj) - 1)

    @classmethod
    def from_parents(cls, site_prev, site_next, index_t):
        if not np.isclose(site_prev.time_at_index(index_t), site_next.time_at_index(index_t)):
            raise ValueError('Sites not sync in time for child creation')
        if site_prev.in_obs_at(index_t) and site_next.in_obs_at(index_t):
            raise ValueError('Illegal child creation between two obstacle-captured sites')
        costate_prev = site_prev.costate_at_index(index_t)
        costate_next = site_next.costate_at_index(index_t)
        if site_prev.in_obs_at(index_t):
            costate_prev = -site_prev.control_at_index(index_t) * np.linalg.norm(costate_next)
        if site_next.in_obs_at(index_t):
            costate_next = -site_next.control_at_index(index_t) * np.linalg.norm(costate_prev)
        name = Site.name_from_parents_name(site_prev.name, site_next.name)
        return Site(site_prev.time_at_index(index_t), index_t,
                    0.5 * (site_prev.state_at_index(index_t) + site_next.state_at_index(index_t)),
                    0.5 * (costate_prev + costate_next),
                    0.5 * (site_prev.cost_at_index(index_t) + site_next.cost_at_index(index_t)),
                    site_prev.n_time,
                    name=name)

    @property
    def name_display(self):
        return self.name[:3] + ('' if len(self.name) == 3 else ('-' + to_alpha(int(self.name[3:][::-1], 2))))

    @classmethod
    def build_prefix(cls, i: int):
        return to_alpha(i).rjust(3, 'A')

    @classmethod
    def index_from_prefix(cls, prefix: str):
        if len(prefix) > 3:
            raise ValueError('Prefix exceeds length 3')
        return (ord(prefix[0]) - 65) * 676 + (ord(prefix[1]) - 65) * 26 + (ord(prefix[2]) - 65)

    @classmethod
    def next_prefix(cls, prefix: str, n_total_prefix: int):
        id_prf = cls.index_from_prefix(prefix)
        return cls.build_prefix((id_prf + 1) % n_total_prefix)

    @classmethod
    def name_from_parents_name(cls, name_prev: str, name_next: str):
        suff_prev = name_prev[3:]
        suff_next = name_next[3:]
        if name_prev[:3] != name_next[:3]:
            suffix = suff_prev + '1'
        else:
            depth = max(len(suff_prev), len(suff_next)) + 1
            suffix = bin((int(suff_prev.ljust(depth, '0'), 2) +
                          int(suff_next.ljust(depth, '0'), 2)) // 2)[2:].rjust(depth, '0')
        return name_prev[:3] + suffix

    @classmethod
    def parents_name_from_name(cls, name: str, n_total_prefix: int) -> tuple[Optional[str], Optional[str]]:
        prefix = name[:3]
        suffix = name[3:]
        if len(suffix) == 0:
            return None, None
        depth = len(suffix)
        suffix_next = bin(int(suffix, 2) + 1)[2:].rjust(depth, '0')
        if len(suffix_next) > depth:
            name_next = cls.next_prefix(prefix, n_total_prefix)
        else:
            name_next = prefix + suffix_next.rstrip('0')
        id_suffix_prev = int(suffix, 2) - 1
        if id_suffix_prev == 0:
            name_prev = prefix
        else:
            suffix_prev = bin(id_suffix_prev)[2:].rjust(depth, '0')
            name_prev = prefix + suffix_prev.rstrip('0')
        return name_prev, name_next

    @property
    def n_time(self):
        return len(self.next_nb)

    def in_obs_at(self, index: int):
        if len(self.obstacle_name) == 0:
            return False
        if index >= self.index_t_obs:
            return True
        return False

    def _update_obstacle_info(self):
        obs_events = [k for k, v in self.traj.events.items() if k.startswith('obs_') and v.shape[0] > 0]
        if len(obs_events) > 0:
            # Assuming at most one obstacle
            obs_name = obs_events[0]
            obs_time = self.traj.events[obs_name][0]
            self.obstacle_name = obs_name
            self.index_t_obs = self.index_t_init + np.searchsorted(self.traj.times, obs_time)

    def extend_traj(self, traj: Trajectory):
        if self.traj is None:
            self.traj = traj
        else:
            self.traj = self.traj + traj
            if self.traj.times.shape[0] >= 2:
                cond = np.all(np.isclose(self.traj.times[1:] - self.traj.times[:-1],
                                         self.traj.times[1] - self.traj.times[0]))
                assert cond
        self._update_obstacle_info()

    def init_next_nb(self, site):
        self.next_nb[0] = site

    def time_at_index(self, index: int):
        return self.traj.times[index - self.index_t_init]

    def state_at_index(self, index: int):
        return self.traj.states[index - self.index_t_init]

    def costate_at_index(self, index: int):
        return self.traj.costates[index - self.index_t_init]

    def state_aug_at_index(self, index: int):
        return np.concatenate(
            (self.traj.states[index - self.index_t_init], self.traj.costates[index - self.index_t_init]))

    def control_at_index(self, index: int):
        return self.traj.controls[index - self.index_t_init]

    def cost_at_index(self, index: int):
        return self.traj.cost[index - self.index_t_init]

    def is_root(self):
        return len(self.name) == 3

    def connect_to_parents(self, sites_dict, n_total_prefix):
        name_prev, name_next = Site.parents_name_from_name(self.name, n_total_prefix)
        site_prev, site_next = sites_dict[name_prev], sites_dict[name_next]
        assert (site_prev.index_t_check_next in [self.index_t_init - 1, self.index_t_init])
        site_prev.next_nb[self.index_t_init] = self
        site_prev.index_t_check_next = self.index_t_init
        self.next_nb[self.index_t_init] = site_next

    def integrate(self, t_eval: ndarray):
        pass

    def extrapolate_back_traj(self, sites_dict, n_total_prefix):
        name_prev, name_next = Site.parents_name_from_name(self.name, n_total_prefix)
        coeff_prev, coeff_next = 0.5, 0.5
        index_hi = self.index_t_init - 1
        times = self.traj.times.copy()
        states = self.traj.states.copy()
        n = times.shape[0]
        costates = self.traj.costates.copy() if self.traj.costates is not None else np.nan * np.ones((n, 2))
        controls = self.traj.controls.copy() if self.traj.controls is not None else np.nan * np.ones((n, 2))
        costs = self.traj.cost.copy()
        while name_prev is not None and name_next is not None:
            site_prev, site_next = sites_dict[name_prev], sites_dict[name_next]
            index_lo = max(site_prev.index_t_init, site_next.index_t_init)
            rec_prev = site_prev.index_t_init > site_next.index_t_init
            s_prev = slice(index_lo - site_prev.index_t_init, index_hi - site_prev.index_t_init + 1)
            s_next = slice(index_lo - site_next.index_t_init, index_hi - site_next.index_t_init + 1)
            times = np.concatenate((site_prev.traj.times[s_prev], times))
            states = np.concatenate((
                coeff_prev * site_prev.traj.states[s_prev] + coeff_next * site_next.traj.states[s_next], states))
            n = s_prev.stop - s_prev.start
            costates_prev = np.nan * np.ones((n, 2)) if site_prev.traj.costates is None else \
                site_prev.traj.costates[s_prev]
            costates_next = np.nan * np.ones((n, 2)) if site_next.traj.costates is None else \
                site_next.traj.costates[s_next]
            costates = np.concatenate((coeff_prev * costates_prev + coeff_next * costates_next,
                                       costates))
            controls_prev = np.nan * np.ones((n, 2)) if site_prev.traj.controls is None else \
                site_prev.traj.controls[s_prev]
            controls_next = np.nan * np.ones((n, 2)) if site_next.traj.controls is None else \
                site_next.traj.controls[s_next]
            controls = np.concatenate((coeff_prev * controls_prev + coeff_next * controls_next,
                                       controls))
            costs = np.concatenate((coeff_prev * site_prev.traj.cost[s_prev] + coeff_next * site_next.traj.cost[s_next],
                                    costs))
            if index_lo == 0:
                break
            index_hi = index_lo - 1
            if rec_prev:
                name_prev = Site.parents_name_from_name(name_prev, n_total_prefix)[0]
                coeff_prev = coeff_prev / 2
                coeff_next = 1 - coeff_prev
            else:
                name_next = Site.parents_name_from_name(name_next, n_total_prefix)[1]
                coeff_next = coeff_next / 2
                coeff_prev = 1 - coeff_next
        # TODO: validate this
        self.traj_full = Trajectory(times, states, self.traj.coords, controls=controls, costates=costates, cost=costs,
                                    events=self.traj.events)

    def __str__(self):
        return f"<Site {self.name}>"

    def __repr__(self):
        return self.__str__()


class SolverEF(ABC):
    _ALL_MODES = ['time', 'energy']

    def __init__(self, pb: NavigationProblem,
                 total_duration: float,
                 mode: str = 'time',
                 max_depth: int = 20,
                 n_time: int = 100,
                 n_costate_angle: int = 30,
                 t_init: Optional[float] = None,
                 n_costate_norm: int = 10,
                 costate_norm_bounds: tuple[float] = (0., 1.),
                 target_radius: Optional[float] = None,
                 abs_max_step: Optional[float] = None,
                 rel_max_step: Optional[float] = None):
        if mode not in self._ALL_MODES:
            raise ValueError('Mode %s is not defined' % mode)
        if mode == 'energy':
            raise ValueError('Mode "energy" is not implemented yet')
        # Problem is assumed to be well conditioned ! (non-dimensionalized)
        self.pb = pb
        # Same for time
        self.total_duration = total_duration
        self.t_init = t_init if t_init is not None else self.pb.model.ff.t_start
        self.target_radius = target_radius if target_radius is not None else pb.target_radius
        self._target_radius_sq = self.target_radius ** 2
        self.times: ndarray = np.linspace(self.t_init, self.t_upper_bound, n_time)
        self.costate_norm_bounds = costate_norm_bounds
        self.n_costate_angle = n_costate_angle
        self.n_costate_norm = n_costate_norm
        self.max_depth = max_depth
        self.success = False
        self.depth = 0
        self.traj_groups: list[tuple[Trajectory]] = []
        self._id_optitraj: int = 0
        self.events = {'target': self._event_target}
        self.obstacles = {}
        classes = {}
        for obs in self.pb.obstacles:
            name = obs.__class__.__name__
            n = classes.get(name)
            if n is None:
                classes[name] = 1
            else:
                classes[name] += 1
            full_name = 'obs_' + name + '_' + str(n if n is not None else 0)
            self.events[full_name] = obs.event
            self.obstacles[full_name] = obs
        self.obs_active = None
        self.obs_active_trigo = True
        abs_max_step = self.total_duration if abs_max_step is None else abs_max_step
        rel_max_step = 5e-3 if rel_max_step is None else rel_max_step
        self.max_int_step: Optional[float] = min(abs_max_step, rel_max_step * self.total_duration)

        self.dyn_augsys = self.dyn_augsys_cartesian if self.pb.coords == Utils.COORD_CARTESIAN else self.dyn_augsys_gcs

    def setup(self):
        self.traj_groups = []

    @property
    def trajs(self) -> list[Trajectory]:
        return []

    @property
    def t_upper_bound(self):
        return self.t_init + self.total_duration

    def dyn_augsys(self, t: float, y: ndarray):
        pass

    def dyn_augsys_cartesian(self, t: float, y: ndarray):
        return self.pb.augsys_dyn_timeopt_cartesian(t, y[:2], y[2:])

    def dyn_augsys_gcs(self, t: float, y: ndarray):
        return self.pb.augsys_dyn_timeopt_gcs(t, y[:2], y[2:])

    def dyn_constr(self, t: float, x: ndarray):
        sign = (2. * self.obs_active_trigo - 1.)
        d = np.array(((0., -sign), (sign, 0.))) @ self.obs_active.d_value(x)
        ff_val = self.pb.model.ff.value(t, x)
        return ff_val + directional_timeopt_control(ff_val, d, self.pb.srf_max)

    @non_terminal
    def _event_target(self, _, x):
        return np.sum(np.square(x[:2] - self.pb.x_target)) - self._target_radius_sq

    @terminal
    def _event_quit_obs(self, t, x):
        n = self.obs_active.d_value(x) / np.linalg.norm(self.obs_active.d_value(x))
        ff_val = self.pb.model.ff.value(t, x)
        ff_ortho = ff_val @ n
        return self.pb.srf_max - np.abs(ff_ortho)

    def t_events_to_dict(self, t_events: list[ndarray]) -> Dict[str, ndarray]:
        d = {}
        for i, name in enumerate(self.events.keys()):
            d[name] = t_events[i]
        return d

    @property
    def n_time(self):
        return self.times.shape[0]

    def get_trajs(self, depth: int) -> Dict[str, Trajectory]:
        res = {}
        for traj in self.traj_groups[depth]:
            res[str(self.trajs.index(traj))] = traj
        return res

    def cost_map(self, nx: int = 100, ny: int = 100) -> ndarray:
        res = np.nan * np.ones((nx, ny))
        bl, spacings = self.pb.get_grid_params(nx, ny)
        for traj in self.trajs:
            for k, state in enumerate(traj.states):
                position = (state - bl) / spacings
                index = tuple(
                    np.clip(np.floor(position).astype(np.int32), np.array((0, 0)), np.array((nx - 1, ny - 1))))
                if np.isnan(res[index]) or res[index] > traj.times[k]:
                    res[index] = traj.times[k]
        return res

    def save_results(self):
        self.pb.io.save_trajs(self.trajs, group_name='ef_01')
        self.pb.io.save_ff(self.pb.model.ff, bl=self.pb.bl, tr=self.pb.tr)


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
                                  np.array(tuple(self.pb.x_init) + tuple(costate)), t_eval=t_eval,
                                  events=list(self.events.values()))
            traj = Trajectory.cartesian(res.t, res.y.transpose()[:, :2], costates=res.y.transpose()[:, 2:],
                                        events=self.t_events_to_dict(res.t_events), cost=res.t - self.t_init)
            if traj.events['target'].shape[0] > 0:
                self.success = True
            traj_group.append(traj)
        self.traj_groups.append(tuple(traj_group))
        self.layers.append(tuple(angles))
        self.depth += 1

    @property
    def trajs(self) -> list[Trajectory]:
        res = []
        for group in self.traj_groups:
            res.extend(list(group))
        return res


class SolverEFResampling(SolverEF):

    def __init__(self, *args, max_dist: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_dist = max_dist if max_dist is not None else self.target_radius
        self._max_dist_sq = self.max_dist ** 2
        self.sites: dict[str, Site] = {}
        self._sites_by_depth: list[list[Site]] = [[] for _ in range(self.max_depth)]
        self._to_shoot_sites: list[Site] = []
        self.solution_sites: set[Site] = set()
        self.solution_site: Optional[Site] = None

    def setup(self):
        super().setup()
        self._to_shoot_sites = [
            Site(self.t_init, 0, self.pb.x_init, np.array((np.cos(theta), np.sin(theta))), 0., self.n_time,
                 name=Site.build_prefix(i_theta))
            for i_theta, theta in enumerate(np.linspace(0, 2 * np.pi, self.n_costate_angle, endpoint=False))]
        for i_site, site in enumerate(self._to_shoot_sites):
            site.init_next_nb(self._to_shoot_sites[(i_site + 1) % len(self._to_shoot_sites)])
        self.sites = {site.name: site for site in self._to_shoot_sites}
        self._sites_by_depth[0].extend(list(self.sites.values()))

    @property
    def trajs(self):
        return list(map(lambda x: x.traj, [site for site in self.sites.values() if site.traj is not None]))

    def solve_ivp_constr(self, y0, t_eval, in_obs=False) -> Optional[Trajectory]:
        if t_eval.shape[0] <= 1:
            return None
        if not in_obs:
            res = scitg.solve_ivp(self.dyn_augsys, (t_eval[0], t_eval[-1]), y0,
                                  t_eval=t_eval[1:],  # Remove initial point which is redudant
                                  events=list(self.events.values()), dense_output=True, max_step=self.max_int_step)
            if len(res.t) == 0:
                times = np.array(())
                states = np.array(((), ()))
                costates = np.array(((), ()))
                cost = np.array(())
            else:
                times = res.t
                states = res.y.transpose()[:, :2]
                costates = res.y.transpose()[:, 2:]
                cost = res.t - self.t_init
            events = self.t_events_to_dict(res.t_events)
        else:
            times = np.array(())
            states = np.array(((), ()))
            costates = np.array(((), ()))
            cost = np.array(())
            events = {}
        traj = Trajectory.cartesian(times, states, costates=costates, cost=cost, events=events)

        active_obstacles = [name for name, t_events in traj.events.items()
                            if t_events.shape[0] > 0 and name != 'target']
        times = np.array(())
        states = np.array(((), ()))
        costates = None
        controls = None
        cost = np.array(())
        if len(active_obstacles) >= 2:
            warnings.warn("Multiple active obstacles", category=RuntimeWarning)
        if len(active_obstacles) >= 1:
            obs_name = active_obstacles[0]
            t_enter_obs = traj.events[obs_name][0]
            self.obs_active = self.obstacles[obs_name]
            state_aug_cross = res.sol(t_enter_obs)
            cross = np.cross(self.obs_active.d_value(state_aug_cross[:2]),
                             self.pb.model.dyn.value(t_enter_obs, state_aug_cross[:2],
                                                     -state_aug_cross[2:] / np.linalg.norm(state_aug_cross[2:])))
            self.obs_active_trigo = cross >= 0.
            sign = 2 * self.obs_active_trigo - 1
            direction = np.array(((0, -sign), (sign, 0))) @ self.obs_active.d_value(state_aug_cross[:2])
            if is_possible_direction(self.pb.model.ff.value(t_enter_obs, state_aug_cross[:2]),
                                     direction, self.pb.srf_max):
                res = scitg.solve_ivp(self.dyn_constr, (t_enter_obs, t_eval[-1]), state_aug_cross[:2],
                                      t_eval=t_eval[t_eval > t_enter_obs], events=[self._event_quit_obs],
                                      max_step=self.max_int_step)
                if len(res.t) == 0:
                    times = np.array(())
                    states = np.array(((), ()))
                    costates = np.array(((), ()))
                    controls = None
                    cost = np.array(())
                else:
                    times = res.t
                    states = res.y.transpose()
                    costates = None
                    controls = np.array([self.dyn_constr(t, x) - self.pb.model.ff.value(t, x)
                                         for t, x in zip(res.t, res.y.transpose())])
                    cost = res.t - self.t_init
        traj_obs = Trajectory.cartesian(times, states, costates=costates, controls=controls, cost=cost)
        return traj + traj_obs

    def integrate_site_to_target_time(self, site: Site, t_target: float):
        self.integrate_site_to_target_index(site, self.times.searchsorted(t_target, side='right') - 1)

    def integrate_site_to_target_index(self, site: Site, index_t: int):
        # Assuming site received integration up to site.index_t included
        if site.index_t >= index_t:
            return
        t_start = self.times[site.index_t]
        t_end = self.times[index_t]
        state_start = site.state_at_index(site.index_t)
        costate_start = site.costate_at_index(site.index_t)
        t_eval = np.linspace(t_start, t_end, index_t - site.index_t + 1)
        traj = self.solve_ivp_constr(np.hstack((state_start, costate_start)), t_eval,
                                     in_obs=site.in_obs_at(site.index_t))
        if len(traj) < t_eval.shape[0] - 1:
            site.close("Integration stopped")
            return

        if traj.events.get('target') is not None and traj.events.get('target').shape[0] > 0:
            self.success = True
            self.solution_sites.add(site)

        site.extend_traj(traj)

    def step(self):
        for site in tqdm(self._to_shoot_sites, desc='Depth %d' % self.depth):
            self.integrate_site_to_target_time(site, self.t_upper_bound)
            if site.traj is not None:
                if not site.is_root():
                    site.connect_to_parents(self.sites, self.n_costate_angle)
        self._to_shoot_sites = []
        new_sites = self.compute_new_sites()
        self._to_shoot_sites.extend(new_sites)
        self._sites_by_depth[self.depth].extend(new_sites)
        self.depth += 1

    def solve(self):
        self.setup()
        for _ in range(self.max_depth):
            self.step()
        self.check_solutions()
        for site in self.solution_sites:
            site.extrapolate_back_traj(self.sites, self.n_costate_angle)

    def get_trajs_by_depth(self, depth: int) -> Dict[str, Trajectory]:
        res = {}
        for site in self._sites_by_depth[depth]:
            if site.traj is not None:
                res[site.name_display] = site.traj
        return res

    def get_trajs(self, depth: int) -> Dict[str, Trajectory]:
        # TODO: validate
        res = {}
        for i in range(self.depth):
            res = {**res, **self.get_trajs_by_depth(i)}
        return res

    def site_front(self, index_t):
        site0 = list(self.sites.values())[0]
        l_sites = []
        site = site0.next_nb[index_t]
        while site != site0:
            l_sites.append(site)
            site_nb = site.next_nb[index_t]
            if site_nb is None:
                return site
            site = site_nb
        return l_sites + [site0]

    @property
    def _closed_sites(self):
        return [site for site in self.sites.values() if site.closed]

    @property
    def suboptimal_sites(self):
        return self.solution_sites.difference({self.solution_site})

    def compute_new_sites(self, index_t_hi: Optional[int] = None) -> list[Site]:
        index_t_hi = index_t_hi if index_t_hi is not None else self.n_time - 1
        prev_sites = [site for site in self.sites.values() if not site.closed]
        new_sites: list[Site] = []
        for i_s, site in enumerate(prev_sites):
            site_nb = site.next_nb[site.index_t_check_next]
            new_id_check_next = site.index_t_check_next
            new_site: Optional[Site] = None
            for i in range(site.index_t_check_next, min(site.index_t + 1, site_nb.index_t + 1, index_t_hi)):
                state = site.state_at_index(i)
                state_nb = site_nb.state_at_index(i)
                if site.in_obs_at(i) and site_nb.in_obs_at(i):
                    site.close("Lies in obstacle and neighbour too")
                    break
                if np.sum(np.square(state - state_nb)) > self._max_dist_sq:
                    new_site = Site.from_parents(site, site_nb, i)
                    if len(self.pb._in_obs(new_site.state_at_index(new_site.index_t_init))) > 0:
                        # Choose not to resample points lying within obstacles
                        site.close("Point lying inside obstacle at its creation")
                        new_site = None
                    break
                new_id_check_next = i
            # Update the neighbouring property
            site.next_nb[site.index_t_check_next: new_id_check_next + 1] = \
                [site_nb] * (new_id_check_next - site.index_t_check_next + 1)
            site.index_t_check_next = new_id_check_next
            if new_site is not None:
                self.sites[new_site.name] = new_site
                new_sites.append(new_site)
        return new_sites

    def check_solutions(self):
        for site in self.sites.values():
            if np.any(np.sum(np.square(site.traj.states - self.pb.x_target), axis=-1) < self._target_radius_sq):
                self.success = True
                self.solution_sites.add(site)
        min_cost = None
        for site in self.solution_sites:
            candidate_cost = np.min(
                site.traj.cost[np.sum(np.square(site.traj.states - self.pb.x_target), axis=-1) < self._target_radius_sq])
            if min_cost is None or candidate_cost < min_cost:
                min_cost = candidate_cost
                self.solution_site = site

    def cost_map_triangle(self, nx: int = 100, ny: int = 100):
        return cost_map_triangle(list(self.sites.values()), 0, self.n_time - 1, self.pb.bl, self.pb.tr, nx, ny)


class SolverEFTrimming(SolverEFResampling):

    def __init__(self, *args,
                 n_index_per_subframe: int = 10,
                 trimming_band_width: int = 1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_index_per_subframe: int = n_index_per_subframe
        self._trimming_band_width = trimming_band_width
        self.i_subframe: int = 0
        self.sites_valid: list[set[Site]] = [set() for _ in range(self.n_subframes + 1)]
        self._sites_created_at_subframe: list[set[Site]] = [set()
                                                            for _ in range(self.n_subframes)]
        self.depth = 1
        self._partial_cost_map = None

    def setup(self):
        super().setup()
        self.sites_valid[0] |= set(self.sites.values())
        self._sites_created_at_subframe[0] |= set(self.sites.values())
        self._to_shoot_sites = []

    @property
    def n_subframes(self):
        return self.n_time // self.n_index_per_subframe + (1 if self.n_time % self.n_index_per_subframe > 0 else 0)

    @property
    def index_t_cur_subframe(self):
        return self.index_t_subframe(self.i_subframe)

    @property
    def index_t_next_subframe(self):
        return min(self.index_t_subframe(self.i_subframe + 1), self.n_time)

    def index_t_subframe(self, i: int):
        return i * self.n_index_per_subframe

    def _sites_closed_at_subframe(self, i: int):
        return [site for site in self.sites_valid[i].union(self._sites_created_at_subframe[i]) if site.closed]

    def step(self):
        self._to_shoot_sites.extend(self.sites_valid[self.i_subframe])
        while len(self._to_shoot_sites) > 0:
            while len(self._to_shoot_sites) > 0:
                site = self._to_shoot_sites.pop()
                self.integrate_site_to_target_index(site, self.index_t_next_subframe - 1)
                if site.traj is not None:
                    if not site.is_root() and not site.has_neighbours:
                        site.connect_to_parents(self.sites, self.n_costate_angle)
                cond = site.index_t == self.index_t_next_subframe - 1 or site.closed
                assert cond
            new_sites = self.compute_new_sites(index_t_hi=self.index_t_next_subframe - 1)
            self._to_shoot_sites.extend(new_sites)
            self._sites_by_depth[0].extend(new_sites)  # Compatibility with SolverEFResampling
            self._sites_created_at_subframe[self.i_subframe] |= set(new_sites)
        # TODO: insert trimming here
        # self.sites_valid[self.i_subframe + 1] = ...
        self.trim()
        self.i_subframe += 1

    def trim(self):
        nx, ny = 100, 100
        sites_considered = self.sites_valid[self.i_subframe] | self._sites_created_at_subframe[self.i_subframe]
        i_subframe_start_cm = self.i_subframe + 1 - self._trimming_band_width
        sites_for_cost_map = self._sites_created_at_subframe[self.i_subframe].union(
            *self.sites_valid[i_subframe_start_cm:self.i_subframe + 1])
        cost_map = cost_map_triangle(sites_for_cost_map, self.index_t_subframe(i_subframe_start_cm),
                                     self.index_t_next_subframe - 1,
                                     self.pb.bl, self.pb.tr, nx, ny)
        self._partial_cost_map = cost_map
        bl, spacings = self.pb.get_grid_params(nx, ny)
        new_valid_sites = []
        for site in [site for site in sites_considered if not site.closed]:
            value_opti = Utils.interpolate(cost_map, bl, spacings, site.state_at_index(self.index_t_next_subframe - 1))
            if np.isnan(value_opti):
                new_valid_sites.append(site)
                continue
            if site.cost_at_index(self.index_t_next_subframe - 1) > 1.1 * value_opti:
                site.close('Trimming')
            else:
                new_valid_sites.append(site)
        self.sites_valid[self.i_subframe + 1] |= set(new_valid_sites)

    def solve(self):
        self.setup()
        for _ in range(self.n_subframes):
            print('Subframe %d/%d, Act: %d, Cls: %d (Tot: %d)' %
                  (self.i_subframe, self.n_subframes - 1, len(self.sites_valid[self.i_subframe]),
                   len([site for site in self.sites.values() if site.closed]), len(self.sites)))
            self.step()
        for site in self.solution_sites:
            site.extrapolate_back_traj(self.sites, self.n_costate_angle)


def cost_map_triangle(sites: Iterable[Site], index_t_lo: int, index_t_hi: int, bl: ndarray, tr: ndarray,
                      nx: int, ny: int) -> ndarray:
    res = np.inf * np.ones((nx, ny))
    grid_vectors = np.stack(np.meshgrid(np.linspace(bl[0], tr[0], nx),
                                        np.linspace(bl[1], tr[1], ny),
                                        indexing='ij'), -1)
    for site in sites:
        for index_t in range(max(index_t_lo, site.index_t_init), min(index_t_hi + 1, site.index_t_check_next)):
            site_nb = site.next_nb[index_t]

            # site, index_t    1 o --------- o 3 site, index + 1
            #                    | \   tri1  |
            #                    |    \      |
            #                    | tri2  \   |
            #                    |          \|
            # site_nb, index_t 2 o-----------o 4 site_nb, index_t + 1
            point1 = site.state_at_index(index_t)
            point3 = site.state_at_index(index_t + 1)
            point2 = site_nb.state_at_index(index_t)
            point4 = site_nb.state_at_index(index_t + 1)
            cost1 = site.cost_at_index(index_t)
            cost3 = site.cost_at_index(index_t + 1)
            cost2 = site_nb.cost_at_index(index_t)
            cost4 = site_nb.cost_at_index(index_t + 1)
            tri1_cost = triangle_mask_and_cost(grid_vectors, point1, point3, point4, cost1, cost3, cost4)
            np.minimum(res, tri1_cost, out=res)
            if index_t > 0:
                tri2_cost = triangle_mask_and_cost(grid_vectors, point1, point4, point2, cost1, cost4, cost2)
                np.minimum(res, tri2_cost, out=res)
    return res
