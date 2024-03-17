import json
import math
import os
import signal
import sys
import warnings
from abc import ABC
from enum import Enum
from typing import Optional, Dict, Iterable

import numpy as np
import scipy.integrate as scitg
from numpy import ndarray
from tqdm import tqdm

from dabry.misc import directional_timeopt_control, Utils, triangle_mask_and_cost, non_terminal, to_alpha, \
    is_possible_direction, diadic_valuation, alpha_to_int, Coords, Chrono
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


class DelayedKeyboardInterrupt:
    # From https://stackoverflow.com/questions/842557/
    # how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py

    def __init__(self):
        self.signal_received = False
        self.old_handler = None

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.', file=sys.stderr)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)


class ClosureReason(Enum):
    SUBOPTIMAL = 0
    IMPOSSIBLE_OBS_TRACKING = 1


class NeuteringReason(Enum):
    # When two neighbours enter obstacle and have different directions
    OBSTACLE_SPLITTING = 0
    INDEFINITE_CHILD_CREATION = 1


class Site:

    def __init__(self, t_init: float, index_t_init: int,
                 state_origin: ndarray, costate_origin: ndarray, cost_origin: float, n_time: int,
                 obstacle_name="", index_t_obs: int = 0, t_enter_obs: Optional[float] = None,
                 obs_trigo: bool = True, name="", init_next_nb=None):
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
        self.prev_index_t_check_next = index_t_init
        self.obstacle_name = obstacle_name
        self.index_t_obs = index_t_obs
        self.t_enter_obs: Optional[float] = t_enter_obs
        self.obs_trigo = obs_trigo
        self.closure_reason: Optional[ClosureReason] = None
        self.neutering_reason: Optional[NeuteringReason] = None
        self._index_neutered = -1
        self._index_closed = None
        self.next_nb: list[Optional[Site]] = [None] * n_time
        self.name = name
        if init_next_nb is not None:
            self.init_next_nb(init_next_nb)

    @property
    def depth(self):
        if '-' not in self.name:
            return 0
        _, depth, _ = self.name.split('-')
        return int(depth)

    @property
    def closed(self):
        return self.closure_reason is not None

    def close(self, reason: ClosureReason, index=None):
        self.closure_reason = reason
        self._index_closed = index if index is not None else self.n_time - 1


    @property
    def neutered(self):
        return self.neutering_reason is not None

    def neuter(self, index: int, reason: NeuteringReason):
        if self.neutering_reason is not None:
            return
        self._index_neutered = index
        self.neutering_reason = reason

    @property
    def has_neighbours(self):
        return any([next_nb is not None for next_nb in self.next_nb])

    @property
    def index_t(self):
        return self.index_t_init + (0 if self.traj is None else len(self.traj) - 1)

    @property
    def n_time(self):
        return len(self.next_nb)

    @property
    def in_obs(self):
        return len(self.obstacle_name) > 0

    def in_obs_at(self, index: int):
        if not self.in_obs:
            return False
        if index >= self.index_t_obs:
            return True
        return False

    def extend_traj(self, traj: Trajectory):
        if self.traj is None:
            self.traj = traj
        else:
            self.traj = self.traj + traj
            if self.traj.times.shape[0] >= 2:
                cond = np.all(np.isclose(self.traj.times[1:] - self.traj.times[:-1],
                                         self.traj.times[1] - self.traj.times[0]))
                assert cond

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
        return '-' not in self.name

    def __str__(self):
        return f"<Site {self.name}>"

    def __repr__(self):
        return self.__str__()


class SiteManager:

    def __init__(self, n_sectors: int, max_depth: int, looping_sectors=True):
        self.n_sectors = n_sectors
        self.max_depth = max_depth
        self.looping_sectors = looping_sectors

        # Avoid recomputing several times this value
        self._pow_2_max_depth = pow(2, self.max_depth)
        self._pow_2_max_depth_minus_1 = pow(2, self.max_depth - 1)
        self._n_prefix_chars = int(np.ceil(math.log(n_sectors) / math.log(26)))
        self._n_depth_chars = int(np.ceil(math.log10(self.max_depth)))
        self._n_location_chars = int(np.ceil(math.log10(self._pow_2_max_depth)))

    @property
    def n_total_sites(self):
        return self.n_sectors * self._pow_2_max_depth + (0 if self.looping_sectors else 1)

    def check_pdl(self, prefix: int, depth: int, location: int):
        prefix_bound = self.n_sectors if self.looping_sectors else self.n_sectors + 1
        if prefix < 0 or prefix >= prefix_bound:
            raise IndexError(f'"prefix" out of range. (val: {prefix}, range: (0, {prefix_bound}))')
        if depth < 0 or depth >= self.max_depth + 1:
            raise IndexError(f'"depth" out of range. (val: {depth}, range: (0, {self.max_depth + 1}))')
        if location < 0 or location >= self._pow_2_max_depth_minus_1:
            raise IndexError(f'"location" out of range. (val: {location}, range: (0, {self._pow_2_max_depth_minus_1}))')
        if not self.looping_sectors:
            if prefix == self.n_sectors + 1 and (depth > 0 or location > 0):
                raise IndexError('PDL out of range')

    def index_from_pdl(self, prefix: int, depth: int, location: int):
        self.check_pdl(prefix, depth, location)
        if depth == 0 and location == 0:
            return self._pow_2_max_depth * prefix
        else:
            return self._pow_2_max_depth * prefix + pow(2, self.max_depth - depth) * (2 * location + 1)

    def pdl_from_index(self, index: int):
        prefix = index // self._pow_2_max_depth
        if index % self._pow_2_max_depth == 0:
            return prefix, 0, 0
        else:
            u = index % self._pow_2_max_depth
            depth = self.max_depth - diadic_valuation(u)
            location = ((u // pow(2, self.max_depth - depth)) - 1) // 2
            return prefix, depth, location

    def name_from_index(self, index: int):
        return self.name_from_pdl(*self.pdl_from_index(index))

    def name_from_pdl(self, prefix: int, depth: int, location: int):
        self.check_pdl(prefix, depth, location)
        s_prefix = to_alpha(prefix).rjust(self._n_prefix_chars, 'A')
        if depth == 0 and location == 0:
            return s_prefix
        return "{prefix}-{depth}-{location}".format(
            prefix=s_prefix,
            depth=str(depth),  # .rjust(self._n_depth_chars, '0'),
            location=str(location)  # .rjust(self._n_location_chars, '0')
        )

    @staticmethod
    def pdl_from_name(name: str):
        if '-' not in name:
            return alpha_to_int(name), 0, 0
        s_prefix, s_depth, s_location = name.split('-')
        return alpha_to_int(s_prefix), int(s_depth), int(s_location)

    def index_from_name(self, name: str):
        return self.index_from_pdl(*self.pdl_from_name(name))

    def name_from_parents_name(self, name_prev: str, name_next: str):
        index_prev = self.index_from_name(name_prev)
        index_next = self.index_from_name(name_next)
        if index_next < index_prev:
            if not self.looping_sectors:
                raise ValueError('Next parent index is inferior to previous parent in non-looping mode')
            index_next += self.n_total_sites
        _, d1, _ = self.pdl_from_name(name_prev)
        _, d2, _ = self.pdl_from_name(name_next)
        if d1 == self.max_depth - 1 or d2 == self.max_depth - 1:
            raise Exception('Maximum depth reached: cannot create child site')
        cond = (index_prev + index_next) % 2 == 0
        assert cond
        return self.name_from_index((index_prev + index_next) // 2)

    def parents_name_from_name(self, name: str) -> tuple[Optional[str], Optional[str]]:
        index = self.index_from_name(name)
        prefix, depth, location = self.pdl_from_name(name)
        if depth == 0 and location == 0:
            return None, None
        return self.name_from_index(index - pow(2, self.max_depth - depth)), self.name_from_index(
            (index + pow(2, self.max_depth - depth)) % self.n_total_sites)

    def site_from_parents(self, site_prev, site_next, index_t) -> Site:
        if not np.isclose(site_prev.time_at_index(index_t), site_next.time_at_index(index_t)):
            raise ValueError('Sites not sync in time for child creation')
        # if site_prev.in_obs_at(index_t) and site_next.in_obs_at(index_t):
        #     raise ValueError('Illegal child creation between two obstacle-captured sites')
        costate_prev = site_prev.costate_at_index(index_t)
        costate_next = site_next.costate_at_index(index_t)
        if site_prev.in_obs_at(index_t):
            costate_prev = -site_prev.control_at_index(index_t) * np.linalg.norm(costate_next)
        if site_next.in_obs_at(index_t):
            costate_next = -site_next.control_at_index(index_t) * np.linalg.norm(costate_prev)
        name = self.name_from_parents_name(site_prev.name, site_next.name)
        return Site(site_prev.time_at_index(index_t), index_t,
                    0.5 * (site_prev.state_at_index(index_t) + site_next.state_at_index(index_t)),
                    0.5 * (costate_prev + costate_next),
                    0.5 * (site_prev.cost_at_index(index_t) + site_next.cost_at_index(index_t)),
                    site_prev.n_time,
                    name=name)


class SolverEF(ABC):
    _ALL_MODES = ['time', 'energy']

    def __init__(self, pb: NavigationProblem,
                 total_duration: Optional[float] = None,
                 mode: str = 'time',
                 max_depth: int = 20,
                 n_time: int = 100,
                 n_costate_sectors: int = 30,
                 t_init: Optional[float] = None,
                 n_costate_norm: int = 10,
                 costate_norm_bounds: tuple[float] = (0., 1.),
                 target_radius: Optional[float] = None,
                 abs_max_step: Optional[float] = None,
                 rel_max_step: Optional[float] = 0.01,
                 free_max_step: bool = True,
                 cost_map_shape: Optional[tuple[int, int]] = (100, 100),
                 ivp_solver: str = 'RK45'):
        if mode not in self._ALL_MODES:
            raise ValueError('Mode %s is not defined' % mode)
        if mode == 'energy':
            raise ValueError('Mode "energy" is not implemented yet')
        # Problem is assumed to be well conditioned ! (non-dimensionalized)
        self.pb = pb
        if total_duration is None:
            # Inflation factor needed for particular case when heuristic trajectory
            # is indeed leading to optimal cost
            self.total_duration = 1.1 * self.pb.auto_time_upper_bound()
            if self.total_duration == np.infty:
                self.total_duration = 2 * self.pb.length_reference / self.pb.srf_max
        else:
            self.total_duration = total_duration
        self.t_init = t_init if t_init is not None else self.pb.model.ff.t_start
        self.target_radius = target_radius if target_radius is not None else pb.target_radius
        self._target_radius_sq = self.target_radius ** 2
        self.times: ndarray = np.linspace(self.t_init, self.t_upper_bound, n_time)
        self.costate_norm_bounds = costate_norm_bounds
        self.n_costate_sectors = n_costate_sectors
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
        abs_max_step = self.total_duration / 2 if abs_max_step is None else abs_max_step
        self.max_int_step: Optional[float] = None if free_max_step else abs_max_step if rel_max_step is None else \
            min(abs_max_step, rel_max_step * self.total_duration)

        self._integrator_kwargs = {'method': ivp_solver}
        if self.max_int_step is not None:
            self._integrator_kwargs['max_step'] = self.max_int_step

        self.dyn_augsys = self.dyn_augsys_cartesian if self.pb.coords == Coords.CARTESIAN else self.dyn_augsys_gcs
        self._cost_map = CostMap(self.pb.bl, self.pb.tr, *cost_map_shape)
        self.chrono = Chrono(f'Solving problem "{self.pb.name}"')

    @property
    def computation_duration(self):
        return self.chrono.t_end - self.chrono.t_start

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

    def dyn_constr(self, t: float, x: ndarray, obstacle: str, trigo: bool):
        sign = (2. * trigo - 1.)
        d = np.array(((0., -sign), (sign, 0.))) @ self.obstacles[obstacle].d_value(x)
        ff_val = self.pb.model.ff.value(t, x)
        return ff_val + directional_timeopt_control(ff_val, d, self.pb.srf_max)

    @non_terminal
    def _event_target(self, _, x):
        return np.sum(np.square(x[:2] - self.pb.x_target)) - self._target_radius_sq

    @terminal
    def _event_quit_obs(self, t, x, obstacle: str, trigo: bool):
        d_value = self.obstacles[obstacle].d_value(x)
        n = d_value / np.linalg.norm(d_value)
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

    def cost_map_from_trajs(self, nx: int = 100, ny: int = 100) -> ndarray:
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

    def save_results(self, scale_length: Optional[float] = None, scale_time: Optional[float] = None,
                     bl: Optional[ndarray] = None, time_offset: Optional[float] = None):
        self.pb.io.clean_output_dir()
        self.pb.io.save_trajs(self.trajs, group_name='ef_01', scale_length=scale_length, scale_time=scale_time,
                              bl=bl, time_offset=time_offset)
        self.save_info()
        self.pb.save_ff()
        self.pb.save_obs()
        self.pb.save_info()

    def save_info(self):
        pb_info = {
            'solver': self.__class__.__name__,
            'computation_duration': self.computation_duration,
        }
        with open(os.path.join(self.pb.io.solver_info_fpath), 'w') as f:
            json.dump(pb_info, f, indent=4)

    def pre_solve(self):
        self.chrono.start()

    def post_solve(self):
        self.chrono.stop()


class SolverEFSimple(SolverEF):

    def __init__(self, *args, **kwargs):
        super(SolverEFSimple, self).__init__(*args, **kwargs)
        self.layers: list[tuple[float]] = []
        self._tested_layers: list = []
        self._id_group_optitraj: int = 0

    def setup(self):
        super().setup()
        self.layers = []
        self._tested_layers = []

    def step(self):
        angles = np.linspace(0., 2 * np.pi, 2 ** self.depth * self.n_costate_sectors + 1)
        if self.depth != 0:
            angles = angles[1::2]
        costate_init = np.stack((np.cos(angles), np.sin(angles)), -1)
        traj_group: list[Trajectory] = []
        for costate in tqdm(costate_init):
            t_eval = self.times
            res = scitg.solve_ivp(self.dyn_augsys, (self.t_init, self.t_upper_bound),
                                  np.array(tuple(self.pb.x_init) + tuple(costate)), t_eval=t_eval,
                                  events=list(self.events.values()), **self._integrator_kwargs)
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

    def __init__(self, *args, max_dist: Optional[float] = None, mode_origin: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_dist = max_dist if max_dist is not None else self.target_radius
        self._max_dist_sq = self.max_dist ** 2
        self.sites: dict[str, Site] = {}
        self._to_shoot_sites: list[Site] = []
        self._checked_for_solution: set[Site] = set()
        self.solution_sites: set[Site] = set()
        self.solution_site: Optional[Site] = None
        self.solution_site_min_cost_index: Optional[int] = None
        self.mode_origin: bool = mode_origin
        self.site_mngr: SiteManager = SiteManager(self.n_costate_sectors, self.max_depth)
        self._ff_max_norm = np.max(np.linalg.norm(self.pb.model.ff.values, axis=-1)) if \
            hasattr(self.pb.model.ff, 'values') else None

    def setup(self):
        super().setup()
        self._to_shoot_sites = [
            Site(self.t_init, 0, self.pb.x_init, 1000 * np.array((np.cos(theta), np.sin(theta))), 0., self.n_time,
                 name=self.site_mngr.name_from_pdl(i_theta, 0, 0))
            for i_theta, theta in enumerate(np.linspace(0, 2 * np.pi, self.n_costate_sectors, endpoint=False))]
        for i_site, site in enumerate(self._to_shoot_sites):
            site.init_next_nb(self._to_shoot_sites[(i_site + 1) % len(self._to_shoot_sites)])
        self.sites = {site.name: site for site in self._to_shoot_sites}

    @property
    def trajs(self):
        return list(map(lambda x: x.traj, [site for site in self.sites.values() if site.traj is not None]))

    def integrate_site_to_target_time(self, site: Site, t_target: float):
        self.integrate_site_to_target_index(site, self.times.searchsorted(t_target, side='right') - 1)

    def integrate_site_to_target_index(self, site: Site, index_t: int):
        # Assuming site received integration up to site.index_t included
        # Integrate up to index_t included
        if site.index_t >= index_t:
            return
        t_start = self.times[site.index_t]
        t_end = self.times[index_t]
        state_start = site.state_at_index(site.index_t)
        costate_start = site.costate_at_index(site.index_t)
        t_eval = np.linspace(t_start, t_end, index_t - site.index_t + 1)
        if t_eval.shape[0] <= 1:
            return
        y0 = np.hstack((state_start, costate_start))
        site_index_t_prev = site.index_t
        if not site.in_obs_at(site.index_t):
            res = scitg.solve_ivp(self.dyn_augsys, (t_eval[0], t_eval[-1]), y0,
                                  t_eval=t_eval[1:],  # Remove initial point which is redudant
                                  events=list(self.events.values()), dense_output=True, **self._integrator_kwargs)

            times = res.t if len(res.t) > 0 else np.array(())
            states = res.y.transpose()[:, :2] if len(res.t) > 0 else np.array(((), ()))
            costates = res.y.transpose()[:, 2:] if len(res.t) > 0 else np.array(((), ()))
            cost = res.t - self.t_init if len(res.t) > 0 else np.array(())
            events = self.t_events_to_dict(res.t_events)

            traj_free = Trajectory.cartesian(times, states, costates=costates, cost=cost, events=events)

            active_obstacles = [name for name, t_events in traj_free.events.items()
                                if t_events.shape[0] > 0 and name != 'target']

            if len(active_obstacles) >= 2:
                warnings.warn("Multiple active obstacles", category=RuntimeWarning)
            if len(active_obstacles) >= 1:
                obs_name = active_obstacles[0]
                t_enter_obs = traj_free.events[obs_name][0]
                site.t_enter_obs = t_enter_obs
                site.obstacle_name = obs_name
                site.index_t_obs = site.index_t + len(traj_free) + 1

                state_aug_cross = res.sol(t_enter_obs)
                cross = np.cross(self.obstacles[obs_name].d_value(state_aug_cross[:2]),
                                 self.pb.model.dyn.value(t_enter_obs, state_aug_cross[:2],
                                                         -state_aug_cross[2:] / np.linalg.norm(
                                                             state_aug_cross[2:])))
                site.obs_trigo = cross >= 0.
                sign = 2 * site.obs_trigo - 1
                direction = np.array(((0, -sign), (sign, 0))) @ self.obstacles[obs_name].d_value(state_aug_cross[:2])
                if is_possible_direction(self.pb.model.ff.value(t_enter_obs, state_aug_cross[:2]),
                                         direction, self.pb.srf_max):
                    res = scitg.solve_ivp(self.dyn_constr, (t_enter_obs, t_eval[t_eval > t_enter_obs][0]),
                                          state_aug_cross[:2],
                                          t_eval=np.array((t_eval[t_eval > t_enter_obs][0],)),
                                          args=[site.obstacle_name, site.obs_trigo], **self._integrator_kwargs)

                    times = res.t if len(res.t) > 0 else np.array(())
                    states = res.y.transpose() if len(res.t) > 0 else np.array(((), ()))
                    controls = np.array([
                        self.dyn_constr(t, x, site.obstacle_name, site.obs_trigo) - self.pb.model.ff.value(t, x)
                        for t, x in zip(res.t, res.y.transpose())]) if len(res.t) > 0 else np.array(((), ()))
                    cost = res.t - self.t_init if len(res.t) > 0 else np.array(())

                    traj_singleton = Trajectory.cartesian(times, states, cost=cost, controls=controls)
                    if len(traj_singleton) == 0:
                        site.close(ClosureReason.IMPOSSIBLE_OBS_TRACKING, index=site.index_t + len(traj_free))
                else:
                    traj_singleton = Trajectory.empty()
                    site.close(ClosureReason.IMPOSSIBLE_OBS_TRACKING, index=site.index_t + len(traj_free))
            else:
                traj_singleton = Trajectory.empty()
        else:
            traj_free = Trajectory.empty()
            traj_singleton = Trajectory.empty()
        traj = traj_free + traj_singleton
        site.extend_traj(traj)
        if len(traj_free) > 0 and len(traj_singleton) == 0:
            return

        # Here, either site.index_t == index_t or site.index_t < index_t and obstacle mode is on
        if site.index_t < index_t and len(site.obstacle_name) > 0:
            res = scitg.solve_ivp(self.dyn_constr, (t_eval[site.index_t - site_index_t_prev], t_eval[-1]),
                                  site.state_at_index(site.index_t),
                                  t_eval=t_eval[site.index_t - site_index_t_prev + 1:], events=[self._event_quit_obs],
                                  args=[site.obstacle_name, site.obs_trigo],
                                  **self._integrator_kwargs)
            times = res.t if len(res.t) > 0 else np.array(())
            states = res.y.transpose() if len(res.t) > 0 else np.array(((), ()))
            controls = np.array([
                self.dyn_constr(t, x, site.obstacle_name, site.obs_trigo) - self.pb.model.ff.value(t, x)
                for t, x in zip(res.t, res.y.transpose())]) if len(res.t) > 0 else np.array(((), ()))
            cost = res.t - self.t_init if len(res.t) > 0 else np.array(())

            traj_constr = Trajectory.cartesian(times, states, cost=cost, controls=controls)
            if res.t_events[0].size > 0:
                # The trajectory was unable to follow obstacle
                site.close(ClosureReason.IMPOSSIBLE_OBS_TRACKING, index=site.index_t + len(traj_constr))
        else:
            traj_constr = Trajectory.empty()

        site.extend_traj(traj_constr)

        # if traj.events.get('target') is not None and traj.events.get('target').shape[0] > 0:
        #     self.success = True
        #     self.solution_sites.add(site)

    @property
    def validity_list(self):
        return \
            [
                (site.index_t_check_next, site) for site in
                set(self.sites.values()).difference(
                    set(site for site in self.sites.values() if len(site.traj) == 1)
                )
                if not (site.neutered and site.index_t_check_next + 1 >= site._index_neutered) and
                   not (site.closed and site.index_t_check_next == site.index_t) and
                   not (site.next_nb[site.index_t_check_next].closed and site.index_t_check_next + 1 >=
                        site.next_nb[site.index_t_check_next]._index_closed)
            ]

    @property
    def validity_index(self):
        vlist = self.validity_list
        return 0 if len(vlist) == 0 else min(vlist, key=lambda x: x[0])[0]

    @property
    def nemesis(self):
        vlist = self.validity_list
        if len(vlist) == 0:
            return None
        else:
            return vlist[[a[0] for a in vlist].index(self.validity_index)][1]

    def missed_obstacles(self):
        res = []
        for site in self.sites.values():
            if np.any(np.any(self.pb.in_obs(state)) for state in site.traj.states):
                res.append(site)
        return res

    def step(self):
        for site in tqdm(self._to_shoot_sites, desc='Depth %d' % self.depth, file=sys.stdout):
            self.integrate_site_to_target_time(site, self.t_upper_bound)
            if site.traj is not None:
                if not site.is_root():
                    self.connect_to_parents(site)
        self._to_shoot_sites = []
        new_sites = self.compute_new_sites()
        self._to_shoot_sites.extend(new_sites)
        # self.trim_distance()
        print("Cost guarantee: {val:.3f}/{total:.3f} ({ratio:.1f}%) {candsol}".format(
            val=self.times[self.validity_index],
            total=self.times[-1],
            ratio=100 * self.times[self.validity_index] / self.times[-1],
            candsol=f"Cand. sol. {self.solution_site.cost_at_index(self.solution_site_min_cost_index):.3f}"
            if self.solution_site is not None else ""
        ))
        self.depth += 1

    def trim_distance(self):
        for site in self.sites.values():
            sup_time = np.linalg.norm(site.state_at_index(site.index_t) - self.pb.x_target) / (self.pb.srf_max +
                                                                                               self._ff_max_norm)
            if self.times[-1] - self.times[site.index_t] > sup_time:
                site.close(ClosureReason.SUBOPTIMAL)

    def solve(self):
        self.setup()
        self.pre_solve()
        dki = DelayedKeyboardInterrupt()
        with dki:
            for _ in range(self.max_depth):
                if dki.signal_received:
                    break
                self.step()
                self.check_solutions()
                if self.found_guaranteed_solution:
                    break
        for site in self.solution_sites:
            self.extrapolate_back_traj(site)
        # missed_obs = self.missed_obstacles()
        # TODO: find a way to avoid near zero obstacle detection
        # if len(missed_obs) > 0:
        #     warnings.warn("Missed obstacles in trajectory collection. Consider lowering integration max step size.")
        self.post_solve()

    def get_cost_map(self):
        self._cost_map.update_from_sites(self.sites.values(), 0, self.validity_index)
        return self._cost_map.values

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
        prev_sites = [site for site in self.sites.values()]
        new_sites: list[Site] = []
        for i_s, site in enumerate(prev_sites):
            site_nb = site.next_nb[site.index_t_check_next]
            new_id_check_next = prev_id_check_next = site.index_t_check_next
            new_site: Optional[Site] = None
            upper_index = min(site.index_t + 1, site_nb.index_t + 1, index_t_hi)
            for i in range(site.index_t_check_next, upper_index):
                state = site.state_at_index(i)
                state_nb = site_nb.state_at_index(i)
                # if site.in_obs_at(i) and site_nb.in_obs_at(i):
                #     site.neuter(i, NeuteringReason.SELF_AND_NB_IN_OBS)
                if site.in_obs_at(i) and site_nb.in_obs_at(i) and (site.obs_trigo != site_nb.obs_trigo):
                    site.neuter(i, NeuteringReason.OBSTACLE_SPLITTING)
                if np.sum(np.square(state - state_nb)) > self._max_dist_sq:
                    # Sample from start between free points which are fully defined from origin
                    # Also resample between two points in obstacle if they have the same trigo
                    # else propagate approximation
                    index = 0 if self.mode_origin and \
                                 (not site.in_obs_at(i) and not site_nb.in_obs_at(i) and
                                  site.index_t_init == 0 and site_nb.index_t_init == 0) or \
                                 (site.in_obs_at(i) and site_nb.in_obs_at(i)
                                  and site.index_t_init == 0 and site_nb.index_t_init == 0) else i
                    if site.depth < self.max_depth - 1 and site_nb.depth < self.max_depth - 1 and not site.neutered:
                        # TODO: costate value in obstacle would prevent this neutering
                        try:
                            new_site = self.site_mngr.site_from_parents(site, site_nb, index)
                        except ValueError as e:
                            if not e.args[0].startswith('Costate'):
                                raise e
                            site.neuter(i, NeuteringReason.INDEFINITE_CHILD_CREATION)
                    break
                new_id_check_next = i
            # Update the neighbouring property
            site.next_nb[site.index_t_check_next: new_id_check_next + 1] = \
                [site_nb] * (new_id_check_next - site.index_t_check_next + 1)
            site.prev_index_t_check_next = site.index_t_check_next
            site.index_t_check_next = new_id_check_next
            if new_site is not None:
                self.sites[new_site.name] = new_site
                new_sites.append(new_site)
        return new_sites

    def check_solutions(self):
        to_check = set(self.sites.values()).difference(self._checked_for_solution)
        for site in to_check:
            if np.any(np.sum(np.square(site.traj.states - self.pb.x_target), axis=-1) < self._target_radius_sq):
                self.success = True
                self.solution_sites.add(site)
        min_cost = None
        solutions_sites_cost = {}
        for site in self.solution_sites:
            candidate_cost = np.min(
                site.traj.cost[
                    np.sum(np.square(site.traj.states - self.pb.x_target), axis=-1) < self._target_radius_sq])
            solutions_sites_cost[site] = candidate_cost
            if min_cost is None or candidate_cost < min_cost:
                min_cost = candidate_cost
                self.solution_site = site
                self.solution_site_min_cost_index = self.solution_site.index_t_init + \
                                                    list(self.solution_site.traj.cost).index(min_cost)

        for site, cost in solutions_sites_cost.items():
            if cost > 1.15 * min_cost:
                self.solution_sites.remove(site)

        self._checked_for_solution.union(to_check)

    @property
    def found_guaranteed_solution(self):
        if self.solution_site_min_cost_index is None:
            return False
        return self.validity_index >= self.solution_site_min_cost_index

    def connect_to_parents(self, site: Site):
        name_prev, name_next = self.site_mngr.parents_name_from_name(site.name)
        site_prev, site_next = self.sites[name_prev], self.sites[name_next]
        # assert (site_prev.index_t_check_next in [self.index_t_init - 1, self.index_t_init])
        # Connection to parents
        site_prev.next_nb[site.index_t_init] = site
        site.next_nb[site.index_t_init] = site_next

        # Updating validity of extremal field coverage
        site_prev.index_t_check_next = site.index_t_init

    def extrapolate_back_traj(self, site):
        name_prev, name_next = self.site_mngr.parents_name_from_name(site.name)
        coeff_prev, coeff_next = 0.5, 0.5
        index_hi = site.index_t_init - 1
        times = site.traj.times.copy()
        states = site.traj.states.copy()
        n = times.shape[0]
        costates = site.traj.costates.copy() if site.traj.costates is not None else np.nan * np.ones((n, 2))
        controls = site.traj.controls.copy() if site.traj.controls is not None else np.nan * np.ones((n, 2))
        costs = site.traj.cost.copy()
        while name_prev is not None and name_next is not None:
            site_prev, site_next = self.sites[name_prev], self.sites[name_next]
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
            costs = np.concatenate(
                (coeff_prev * site_prev.traj.cost[s_prev] + coeff_next * site_next.traj.cost[s_next],
                 costs))
            if index_lo == 0:
                break
            index_hi = index_lo - 1
            if rec_prev:
                name_prev = self.site_mngr.parents_name_from_name(name_prev)[0]
                coeff_prev = coeff_prev / 2
                coeff_next = 1 - coeff_prev
            else:
                name_next = self.site_mngr.parents_name_from_name(name_next)[1]
                coeff_next = coeff_next / 2
                coeff_prev = 1 - coeff_next
        site.traj_full = Trajectory(times, states, site.traj.coords, controls=controls, costates=costates,
                                    cost=costs,
                                    events=site.traj.events)

    def save_results(self, scale_length: Optional[float] = None, scale_time: Optional[float] = None,
                     bl: Optional[ndarray] = None, time_offset: Optional[float] = None):
        super(SolverEFResampling, self).save_results(scale_length, scale_time, bl, time_offset)
        if self.solution_site is not None and self.solution_site.traj_full is not None:
            self.pb.io.save_traj(self.solution_site.traj_full, name='solution',
                                 scale_length=scale_length, scale_time=scale_time, bl=bl, time_offset=time_offset)


class SolverEFBisection(SolverEFResampling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SolverEFTrimming(SolverEFResampling):

    def __init__(self, *args,
                 n_index_per_subframe: int = 10,
                 trimming_band_width: int = 3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_index_per_subframe: int = n_index_per_subframe
        self._trimming_band_width = trimming_band_width
        self.i_subframe: int = 0
        self.sites_valid: list[set[Site]] = [set() for _ in range(self.n_subframes + 1)]
        self._sites_created_at_subframe: list[set[Site]] = [set()
                                                            for _ in range(self.n_subframes)]
        self.depth = 1

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
                        self.connect_to_parents(site)
                cond = site.index_t == self.index_t_next_subframe - 1 or site.closed
                assert cond
            new_sites = self.compute_new_sites(index_t_hi=self.index_t_next_subframe - 1)
            self._to_shoot_sites.extend(new_sites)
            self._sites_created_at_subframe[self.i_subframe] |= set(new_sites)
        self.trim()
        self.i_subframe += 1

    def trim(self):
        nx, ny = 100, 100
        sites_considered = self.sites_valid[self.i_subframe] | self._sites_created_at_subframe[self.i_subframe]
        i_subframe_start_cm = self.i_subframe + 1 - self._trimming_band_width
        sites_for_cost_map = self._sites_created_at_subframe[self.i_subframe].union(
            *self.sites_valid[i_subframe_start_cm:self.i_subframe + 1])
        self._cost_map.update_from_sites(sites_for_cost_map, self.index_t_subframe(i_subframe_start_cm),
                                         self.index_t_next_subframe - 1)
        bl, spacings = self.pb.get_grid_params(nx, ny)
        new_valid_sites = []
        for site in [site for site in sites_considered if not site.closed]:
            value_opti = Utils.interpolate(self._cost_map.values, bl, spacings,
                                           site.state_at_index(self.index_t_next_subframe - 1))
            if np.isnan(value_opti):
                new_valid_sites.append(site)
                continue
            if site.cost_at_index(self.index_t_next_subframe - 1) > 1.02 * value_opti:
                site.close(ClosureReason.SUBOPTIMAL, index=self.index_t_next_subframe - 1)
            else:
                new_valid_sites.append(site)
        # TODO: Add distance trimming here
        self.sites_valid[self.i_subframe + 1] |= set(new_valid_sites)

    def solve(self):
        self.setup()
        self.pre_solve()
        for _ in range(self.n_subframes):
            s = 'Subframe {i_subframe}/{n_subframe}, Act: {n_active: >4}, Cls: {n_closed: >4} (Tot: {n_total: >4})'
            print(s.format(
                i_subframe=self.i_subframe,
                n_subframe=self.n_subframes - 1,
                n_active=len(self.sites_valid[self.i_subframe]),
                n_closed=len([site for site in self.sites.values() if site.closed]),
                n_total=len(self.sites))
            )
            self.step()
        self.check_solutions()
        for site in self.solution_sites:
            self.extrapolate_back_traj(site)
        self.post_solve()


class CostMap:

    def __init__(self, bl: ndarray, tr: ndarray, nx: int, ny: int):
        self.bl = bl.copy()
        self.tr = tr.copy()
        self.grid_vectors = np.stack(np.meshgrid(np.linspace(self.bl[0], self.tr[0], nx),
                                                 np.linspace(self.bl[1], self.tr[1], ny),
                                                 indexing='ij'), -1)
        self.values = np.inf * np.ones((nx, ny))

    def update_from_sites(self, sites: Iterable[Site], index_t_lo: int, index_t_hi: int):
        for site in sites:
            # TODO: more efficient screening here
            # lower_index = max(index_t_lo, site.index_t_init, site.prev_index_t_check_next)
            lower_index = max(index_t_lo, site.index_t_init)
            for index_t in range(lower_index, min(index_t_hi + 1, site.index_t_check_next)):
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
                tri1_cost = triangle_mask_and_cost(self.grid_vectors, point1, point3, point4, cost1, cost3, cost4)
                np.minimum(self.values, tri1_cost, out=self.values)
                if index_t > 0:
                    tri2_cost = triangle_mask_and_cost(self.grid_vectors, point1, point4, point2, cost1, cost4, cost2)
                    np.minimum(self.values, tri2_cost, out=self.values)
