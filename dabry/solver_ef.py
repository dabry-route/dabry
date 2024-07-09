import json
import math
import os
import signal
import sys
import warnings
from abc import ABC
from enum import Enum
from typing import Optional, Dict, Iterable, List, Callable

import numpy as np
from numpy import ndarray
from scipy.integrate import solve_ivp, OdeSolution
from scipy.integrate._ivp.ivp import OdeResult
from tqdm import tqdm

from dabry.flowfield import DiscreteFF
from dabry.misc import directional_timeopt_control, Utils, triangle_mask_and_cost, non_terminal, to_alpha, \
    diadic_valuation, alpha_to_int, Chrono, Coords
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

    def __exit__(self, event_type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)


class OdeAugResult:

    def __init__(self, ode_res: OdeResult, events_dict: Dict[str, ndarray],
                 obs_name: Optional[str] = None, obs_trigo: Optional[bool] = None,
                 sol_extra: Optional[Callable] = None):
        self.sol = ode_res.sol
        self.t: ndarray = ode_res.t
        self.y: ndarray = ode_res.y
        self.sol: Optional[OdeSolution] = ode_res.sol
        self.t_events: Optional[List[ndarray]] = ode_res.t_events
        self.y_events: Optional[List[ndarray]] = ode_res.y_events
        self.nfev: int = ode_res.nfev
        self.njev: int = ode_res.njev
        self.nlu: int = ode_res.nlu
        self.status: int = ode_res.status
        self.message: str = ode_res.message
        self.success: bool = ode_res.success

        # New features
        self.events_dict: Dict[str, ndarray] = events_dict
        self.obs_name: Optional[str] = obs_name
        self.obs_trigo: Optional[bool] = obs_trigo
        # Typically an extrapolate of the costate in obstacle mode
        self.sol_extra: Optional[Callable] = sol_extra


class IntStatus(Enum):
    FREE = 0
    OBSTACLE = 1
    OBS_ENTRY = 2
    OBS_EXIT = 3
    FAILURE = 4


class ClosureReason(Enum):
    SUBOPTIMAL = 0
    SUBOPTI_DISTANCE = 1
    IMPOSSIBLE_OBS_TRACKING = 2
    FORCED_OBSTACLE_PENETRATION = 3
    INTEGRATION_FAILED = 4


class NeuteringReason(Enum):
    # When two neighbours enter obstacle and have different directions
    OBSTACLE_SPLITTING = 0
    INTERP_CHILD_IN_OBS = 1


class Site:

    def __init__(self, t_init: float, cost_init: float, state_init: ndarray, costate_init: ndarray,
                 time_grid: ndarray, name: str, t_interp_bounds: Optional[tuple[float, float]] = None,
                 site_prev=None, site_next=None, index_t_check_next: int = 0,
                 data_disc: Optional[ndarray] = None):
        self.ode_legs: List[OdeAugResult] = []
        self.status_int: IntStatus = IntStatus.FREE

        # Initial data can be from problem origin time or interpolation creation time
        self.t_init = t_init
        self.cost_init = cost_init
        self.state_init = state_init
        self.costate_init = costate_init

        self.index_t_check_next = index_t_check_next
        self.prev_index_t_check_next = index_t_check_next
        self.closure_reason: Optional[ClosureReason] = None
        self.neutering_reason: Optional[NeuteringReason] = None
        self.t_closed: Optional[float] = None
        self.t_neutered: Optional[float] = None
        self.time_grid: ndarray = time_grid
        self.next_nb: list[Optional[Site]] = [None] * time_grid.size
        self.name: str = name
        self.t_interp_bounds: Optional[tuple[float, float]] = t_interp_bounds
        self.site_prev = site_prev
        self.site_next = site_next

        self.data_disc: ndarray = np.nan * np.ones((time_grid.size, 5)) if data_disc is None else data_disc
        self._traj: Optional[Trajectory] = None

    @classmethod
    def from_interpolation(cls, site_prev, site_next, t_interp: float, name: str, index_t_check_next: int = 0):
        cost = 0.5 * (site_prev.cost(t_interp) + site_next.cost(t_interp))
        state = 0.5 * (site_prev.state(t_interp) + site_next.state(t_interp))
        costate = 0.5 * (site_prev.costate(t_interp) / np.linalg.norm(site_prev.costate(t_interp)) +
                         site_next.costate(t_interp) / np.linalg.norm(site_next.costate(t_interp)))
        data_disc = 0.5 * (site_prev.data_disc + site_next.data_disc)
        return cls(t_interp, cost, state, costate,
                   site_prev.time_grid, name, (site_prev.time_grid[0], t_interp),
                   site_prev=site_prev, site_next=site_next, index_t_check_next=index_t_check_next, data_disc=data_disc)

    @property
    def traj(self):
        if self._traj is None:
            self._traj = Trajectory(self.time_grid, self.states, costates=self.costates, cost=self.costs)
        return self._traj

    @property
    def interpolated(self):
        return self.t_interp_bounds is not None

    def time_to_index(self, t: float):
        return np.searchsorted(self.time_grid, t, side='right') - 1

    @property
    def index_neutered(self):
        if self.t_neutered is None:
            return -1
        return self.time_to_index(self.t_neutered)

    @property
    def index_closed(self):
        if self.t_closed is None:
            return len(self.time_grid)
        return self.time_to_index(self.t_closed)

    @property
    def t_cur(self):
        if len(self.ode_legs) == 0:
            return self.t_init if not self.interpolated else self.t_interp_bounds[1]
        else:
            return self.ode_legs[-1].sol.t_max

    @property
    def depth(self):
        if '-' not in self.name:
            return 0
        _, depth, _ = self.name.split('-')
        return int(depth)

    @property
    def closed(self):
        return self.closure_reason is not None

    def close(self, reason: ClosureReason):
        self.closure_reason = reason
        self.t_closed = self.t_cur

    @property
    def neutered(self):
        return self.neutering_reason is not None

    def neuter(self, reason: NeuteringReason, t: float):
        if self.neutering_reason is not None:
            return
        self.t_neutered = t
        self.neutering_reason = reason

    @property
    def has_neighbours(self):
        return any([next_nb is not None for next_nb in self.next_nb])

    @property
    def index_t(self):
        return self.time_to_index(self.t_cur)

    @property
    def n_time(self):
        return len(self.next_nb)

    def in_integration_time_bounds(self, t: float):
        if len(self.ode_legs) == 0:
            return False
        if t < self.ode_legs[0].sol.t_min or t > self.ode_legs[-1].sol.t_max:
            return False
        return True

    def in_interpolation_time_bounds(self, t: float):
        return self.t_interp_bounds[0] <= t <= self.t_interp_bounds[1]

    def obs_at(self, t: float):
        if self.interpolated and self.in_interpolation_time_bounds(t):
            # Interpolated trajectory never enters an obstacle
            return None
        if len(self.ode_legs) == 0:
            return None
        ode_leg = self.find_leg(t)
        return ode_leg.obs_name

    def obs_at_index(self, index: int):
        return self.obs_at(self.time_grid[index])

    def in_obs_at_index(self, index: int):
        return self.obs_at_index(index) is not None

    def trigo_at(self, t: float):
        if self.interpolated and self.in_interpolation_time_bounds(t):
            # Interpolated trajectory never enters an obstacle
            return None
        if len(self.ode_legs) == 0:
            return None
        ode_leg = self.find_leg(t)
        return ode_leg.obs_trigo

    def trigo_at_index(self, index: int):
        return self.trigo_at(self.time_grid[index])

    @property
    def obs_cur(self):
        if len(self.ode_legs) == 0:
            return None
        return self.ode_legs[-1].obs_name

    @property
    def trigo_cur(self):
        if len(self.ode_legs) == 0:
            return None
        return self.ode_legs[-1].obs_trigo

    def time_at_index(self, index: int):
        return self.time_grid[index]

    def data_at_index(self, index: int):
        return self.data_disc[index]

    def cost_at_index(self, index: int):
        return self.data_at_index(index)[0]

    def state_at_index(self, index: int):
        return self.data_at_index(index)[1:3]

    def costate_at_index(self, index: int):
        return self.data_at_index(index)[3:5]

    def find_leg(self, t: float):
        if not self.in_integration_time_bounds(t):
            raise ValueError(f'Time out of bounds ({t})')
        if len(self.ode_legs) == 0:
            raise ValueError('Empty leg list')
        for ode_leg in self.ode_legs:
            if ode_leg.sol.t_min <= t <= ode_leg.sol.t_max:
                return ode_leg
        raise ValueError(f'Time not in ODE legs ({t})')

    def data(self, t: float):
        if self.interpolated and self.in_interpolation_time_bounds(t):
            return 0.5 * (self.site_prev.data(t) + self.site_next.data(t))
        leg = self.find_leg(t)
        sol_t = leg.sol(t)
        if sol_t.size <= 3:
            # Costate unfilled, fill it using the extrapolation
            sol_t = np.hstack((sol_t, leg.sol_extra(t)))
        return sol_t

    def cost(self, t: float):
        return self.data(t)[0]

    def state(self, t: float):
        return self.data(t)[1:3]

    def costate(self, t: float):
        return self.data(t)[3:5]

    @property
    def costs(self):
        return self.data_disc[:, 0]

    @property
    def states(self):
        return self.data_disc[:, 1:3]

    @property
    def costates(self):
        return self.data_disc[:, 3:5]

    @property
    def cost_cur(self):
        if len(self.ode_legs) == 0:
            return self.cost_init
        return self.ode_legs[-1].sol(self.ode_legs[-1].sol.t_max)[0]

    @property
    def state_cur(self):
        if len(self.ode_legs) == 0:
            return self.state_init
        return self.ode_legs[-1].sol(self.ode_legs[-1].sol.t_max)[1:3]

    @property
    def costate_cur(self):
        if len(self.ode_legs) == 0:
            return self.costate_init
        return self.ode_legs[-1].sol(self.ode_legs[-1].sol.t_max)[3:5]

    def add_leg(self, ode_aug_res):
        if ode_aug_res.status == -1:
            self.close(ClosureReason.INTEGRATION_FAILED)
            self.status_int = IntStatus.FAILURE
            return
        self.ode_legs.append(ode_aug_res)
        if len(ode_aug_res.t) > 0:
            i_min, i_max = self.time_to_index(ode_aug_res.t[0]), self.time_to_index(ode_aug_res.t[-1]) + 1
            if ode_aug_res.obs_name is None:
                self.data_disc[i_min:i_max, :] = ode_aug_res.y.transpose()
            else:
                self.data_disc[i_min:i_max, :3] = ode_aug_res.y.transpose()
        self.update_status()

    def update_status(self):
        if len(self.ode_legs) == 0:
            return
        last_leg = self.ode_legs[-1]
        if last_leg.obs_name is None:
            # Last leg is a free leg, see if it entered an obstacle
            active_obstacles = [name for name, t_events in last_leg.events_dict.items() if t_events.shape[0] > 0]
            if len(active_obstacles) >= 2:
                warnings.warn("Multiple active obstacles", category=RuntimeWarning)
            if len(active_obstacles) == 1:
                self.status_int = IntStatus.OBS_ENTRY
            else:
                self.status_int = IntStatus.FREE
        else:
            # Last leg is a captive leg, see if it exits
            exit_events = [name for name, t_events in last_leg.events_dict.items() if t_events.size > 0]
            if len(exit_events) > 0:
                self.status_int = IntStatus.OBS_EXIT
            else:
                self.status_int = IntStatus.OBSTACLE

    def is_root(self):
        return '-' not in self.name

    def __str__(self):
        return f"<Site {self.name}>"

    def __repr__(self):
        return self.__str__()


class SiteManager:

    def __init__(self, n_sectors: int, max_depth: int, time_grid: ndarray, looping_sectors=True):
        self.n_sectors = n_sectors
        self.max_depth = max_depth
        self.time_grid: ndarray = time_grid
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

    def site_from_mean_init_cond(self, site_prev: Site, site_next: Site, index: int) -> Site:
        name = self.name_from_parents_name(site_prev.name, site_next.name)
        site_child = Site(site_prev.t_init,
                          0.5 * (site_prev.cost_init + site_next.cost_init),
                          0.5 * (site_prev.state_init + site_next.state_init),
                          0.5 * (site_prev.costate_init + site_next.costate_init),
                          self.time_grid, name, index_t_check_next=index)
        return site_child

    def site_from_interpolation(self, site_prev: Site, site_next: Site, index: int) -> Site:
        name = self.name_from_parents_name(site_prev.name, site_next.name)
        return Site.from_interpolation(site_prev, site_next, site_prev.time_grid[index], name, index_t_check_next=index)


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
            if self.total_duration == np.inf:
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
        self._id_optitraj: int = 0
        self.events = {}
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

        self._cost_map = CostMap(self.pb.bl, self.pb.tr, *cost_map_shape)
        self.chrono = Chrono(f'Solving problem "{self.pb.name}"')

    @property
    def computation_duration(self):
        return self.chrono.t_end - self.chrono.t_start

    def setup(self):
        pass

    @property
    def trajs(self) -> list[Trajectory]:
        return []

    @property
    def t_upper_bound(self):
        return self.t_init + self.total_duration

    def dyn_augsys(self, t: float, y: ndarray):
        return np.hstack((1., self.pb.augsys_dyn_timeopt(t, y[1:3], y[3:5])))

    def dyn_constr(self, t: float, x: ndarray, obstacle: str, trigo: bool, ff_tweak=1.):
        sign = 2 * trigo - 1
        d = np.array(((0., -sign), (sign, 0.))) @ self.obstacles[obstacle].d_value(x[1:3])
        ff_val = ff_tweak * self.pb.model.ff.value(t, x[1:3])
        return np.hstack((1., ff_val + directional_timeopt_control(ff_val, d, self.pb.srf_max)))

    @non_terminal
    def _event_target(self, _, x):
        return np.sum(np.square(x[:2] - self.pb.x_target)) - self._target_radius_sq

    @terminal
    def _event_exit_obs(self, t, x, obstacle: str, _: bool):
        d_value = self.obstacles[obstacle].d_value(x[1:3])
        n = d_value / np.linalg.norm(d_value)
        ff_val = self.pb.model.ff.value(t, x[1:3])
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


class SolverEFResampling(SolverEF):

    def __init__(self, *args,
                 max_dist: Optional[float] = None,
                 mode_resamp_interp: bool = False,
                 mode_obs_stop: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_dist = max_dist if max_dist is not None else self.target_radius
        self._max_dist_sq = self.max_dist ** 2
        self.sites: dict[str, Site] = {}
        self.initial_sites: list[Site] = []
        self._to_shoot_sites: list[Site] = []
        self._checked_for_solution: set[Site] = set()
        self.solution_sites: set[Site] = set()
        self.solution_site: Optional[Site] = None
        self.solution_site_min_cost_index: Optional[int] = None
        self.site_mngr: SiteManager = SiteManager(self.n_costate_sectors, self.max_depth, self.times)
        ff_values = self.pb.model.ff.values if hasattr(self.pb.model.ff, 'values') else \
            DiscreteFF.from_ff(self.pb.model.ff, (self.pb.bl, self.pb.tr), nx=50, ny=50, nt=20, no_diff=True).values
        self._ff_max_norm = np.max(np.linalg.norm(ff_values, axis=-1))
        self.distance_inflat = 1 if self.pb.coords == Coords.CARTESIAN else \
            np.cos(np.max((np.abs(self.pb.bl[1]), np.abs(self.pb.tr[1]))))
        self.mode_resamp_interp = mode_resamp_interp
        self.mode_obs_stop = mode_obs_stop

    def create_initial_sites(self):
        self.initial_sites = [
            Site(self.t_init, 0, self.pb.x_init, np.array((np.cos(theta), np.sin(theta))), self.times,
                 self.site_mngr.name_from_pdl(i_theta, 0, 0))
            for i_theta, theta in enumerate(np.linspace(0, 2 * np.pi, self.n_costate_sectors, endpoint=False))
        ]
        self.sites = {site.name: site for site in self.initial_sites}
        for i_site, site in enumerate(self.initial_sites):
            site.next_nb[0] = self.initial_sites[(i_site + 1) % len(self.initial_sites)]
        return self.initial_sites

    @property
    def trajs(self):
        return list(map(lambda x: x.traj, [site for site in self.sites.values()]))

    def integrate_site_to_target_time(self, site: Site, t_target: float):
        if site.closed:
            raise ValueError('Cannot integrate closed site')
        while not np.isclose(site.t_cur, t_target):
            if site.status_int == IntStatus.FREE:
                y0 = np.hstack((site.cost_cur, site.state_cur, site.costate_cur))
                ode_aug_res = self.integrate_free(site.t_cur, t_target, y0)
                site.add_leg(ode_aug_res)
            elif site.status_int == IntStatus.OBS_ENTRY:
                last_leg = site.ode_legs[-1]
                active_obstacles = [name for name, t_events in last_leg.events_dict.items() if t_events.shape[0] > 0]
                assert (len(active_obstacles) == 1)
                obs_name = active_obstacles[0]
                t_enter_obs = last_leg.events_dict[obs_name][0]
                state_cur, costate_cur = site.state_cur, site.costate_cur
                grad_obs = self.obstacles[obs_name].d_value(state_cur)
                cross = np.cross(grad_obs,
                                 self.pb.model.dyn.value(t_enter_obs, state_cur,
                                                         self.pb.timeopt_control(state_cur, costate_cur)))
                trigo = cross >= 0.
                dot = np.dot(grad_obs, self.dyn_constr(site.t_cur, np.hstack((site.cost_cur, state_cur)),
                                                       obs_name, trigo)[1:3])
                if not np.isclose(dot, 0) or self.mode_obs_stop:
                    site.close(ClosureReason.IMPOSSIBLE_OBS_TRACKING)
                    break
                x0 = np.hstack((site.cost_cur, state_cur))
                ode_aug_res = self.integrate_captive(site.t_cur, t_target, x0, obs_name, trigo)
                site.add_leg(ode_aug_res)
            elif site.status_int == IntStatus.OBSTACLE:
                x0 = np.hstack((site.cost_cur, site.state_cur))
                ode_aug_res = self.integrate_captive(site.t_cur, t_target, x0, site.obs_cur, site.trigo_cur)
                site.add_leg(ode_aug_res)
            elif site.status_int == IntStatus.OBS_EXIT:
                gv = self.dyn_constr(site.t_cur, np.hstack((site.cost_cur, site.state_cur)),
                                     site.obs_cur, site.trigo_cur)[1:3]
                control_cur = gv - self.pb.model.ff.value(site.t_cur, site.state_cur)
                # If trajectory shall enter obstacle because of a flow field unbearable increase
                # then at the given point the constr dyn where the flow field is scaled by a small
                # factor shall have a small component inside the obstacle
                gv_tweaked = self.dyn_constr(site.t_cur, np.hstack((site.cost_cur, site.state_cur)),
                                             site.obs_cur, site.trigo_cur, ff_tweak=1.01)[1:3]
                if ls_float(np.dot(self.obstacles[site.obs_cur].d_value(site.state_cur), gv_tweaked), 0):
                    site.close(ClosureReason.FORCED_OBSTACLE_PENETRATION)
                    break
                costate_artificial = - control_cur / np.linalg.norm(control_cur)
                y0 = np.hstack((site.cost_cur, site.state_cur, costate_artificial))
                ode_aug_res = self.integrate_free(site.t_cur, t_target, y0)
                site.add_leg(ode_aug_res)
            elif site.status_int == IntStatus.FAILURE:
                break

    def integrate_free(self, t_start: float, t_end: float, y0: ndarray):
        res: OdeResult = solve_ivp(self.dyn_augsys, (t_start, t_end), y0,
                                   t_eval=self.times[np.logical_and(t_start <= self.times, self.times <= t_end)],
                                   events=list(self.events.values()), dense_output=True, **self._integrator_kwargs)
        events_dict = self.t_events_to_dict(res.t_events)
        return OdeAugResult(res, events_dict)

    def integrate_captive(self, t_start: float, t_end: float, x0: ndarray, obs_name: str, trigo: bool):
        res: OdeResult = solve_ivp(self.dyn_constr, (t_start, t_end), x0,
                                   t_eval=self.times[np.logical_and(t_start <= self.times, self.times <= t_end)],
                                   events=[self._event_exit_obs], args=[obs_name, trigo],
                                   dense_output=True, **self._integrator_kwargs)
        events_dict = {"exit_forced": res.t_events[0]}
        # Represents the costate as opposite of control vector, norm does not matter
        sol_extra = lambda t: -(self.dyn_constr(t, res.sol(t), obs_name, trigo)[1:3] -
                                self.pb.model.ff.value(t, res.sol(t)[1:3]))
        return OdeAugResult(res, events_dict, obs_name, trigo, sol_extra)

    @property
    def validity_list(self):
        # The minus one in indexes is because closure/neutering can happen simultaneously with distance violation
        # and in this case the check index is stuck -1 behind the closure/neutering index
        return \
            [
                (site.index_t_check_next, site) for site in self.sites_non_void
                if not (site.neutered and site.index_t_check_next >= site.index_neutered - 1) and
                   not (site.closed and site.index_t_check_next == site.index_t) and
                   not (site.next_nb[site.index_t_check_next].closed and site.index_t_check_next >=
                        site.next_nb[site.index_t_check_next].index_closed - 1)
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

    def step(self):
        self._to_shoot_sites = self.feed_new_sites()
        for site in tqdm(self._to_shoot_sites, desc='Depth %d' % self.depth, file=sys.stdout):
            self.integrate_site_to_target_time(site, self.t_upper_bound)
        self._to_shoot_sites = []
        print("Cost guarantee: {val:.3f}/{total:.3f} ({ratio:.1f}%, {valindex}/{totindex}) {candsol}".format(
            val=self.times[self.validity_index],
            total=self.times[-1],
            ratio=100 * self.times[self.validity_index] / self.times[-1],
            valindex=self.validity_index,
            totindex=len(self.times),
            candsol=f"Cand. sol. {self.solution_site.cost_at_index(self.solution_site_min_cost_index):.3f}"
            if self.solution_site is not None else ""
        ))
        self.depth += 1

    def solve(self):
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
        self.post_solve()

    @property
    def sites_non_void(self):
        return [site for site in self.sites.values() if len(site.ode_legs) > 0]

    def get_cost_map(self):
        self._cost_map.update_from_sites(self.sites_non_void, 0, self.validity_index)
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

    def feed_new_sites(self, index_t_hi: Optional[int] = None) -> list[Site]:
        if self.depth == 0:
            new_sites = self.create_initial_sites()
            return new_sites
        else:
            new_sites = self.create_new_sites(index_t_hi)
            for site in new_sites:
                self.connect_site(site)
            return new_sites

    def create_new_sites(self, index_t_hi: Optional[int] = None) -> list[Site]:
        index_t_hi = index_t_hi if index_t_hi is not None else self.n_time - 1
        prev_sites = [site for site in self.sites.values()]
        new_sites: list[Site] = []
        for i_s, site in enumerate(prev_sites):
            site_nb = site.next_nb[site.index_t_check_next]
            new_id_check_next = site.index_t_check_next
            new_site: Optional[Site] = None
            upper_index = min(site.index_t + 1, site_nb.index_t + 1, index_t_hi + 1)
            for i in range(site.index_t_check_next, upper_index):
                state = site.state_at_index(i)
                state_nb = site_nb.state_at_index(i)
                distance_crit = np.sum(np.square(state - state_nb)) <= self._max_dist_sq
                if distance_crit:
                    new_id_check_next = i
                if site.in_obs_at_index(i) and site_nb.in_obs_at_index(i) and \
                        site.obs_at_index(i) == site_nb.obs_at_index(i) and \
                        (site.trigo_at_index(i) != site_nb.trigo_at_index(i)):
                    site.neuter(NeuteringReason.OBSTACLE_SPLITTING, self.times[i])
                    break
                if not distance_crit and site.depth < self.max_depth - 1 and site_nb.depth < self.max_depth - 1 and \
                        not site.neutered:
                    new_site = self.binary_resample(site, site_nb, i)
                    if self.pb.in_obs_tol(new_site.state_cur):
                        new_site = None
                        site.neuter(NeuteringReason.INTERP_CHILD_IN_OBS, self.times[i])
                    break
            # Update the neighbouring property
            site.next_nb[site.index_t_check_next: new_id_check_next + 1] = \
                [site_nb] * (new_id_check_next - site.index_t_check_next + 1)
            site.prev_index_t_check_next = site.index_t_check_next
            site.index_t_check_next = new_id_check_next
            if new_site is not None:
                self.sites[new_site.name] = new_site
                new_sites.append(new_site)
        return new_sites

    def connect_site(self, site: Site):
        name_prev, name_next = self.site_mngr.parents_name_from_name(site.name)
        site_prev, site_next = self.sites[name_prev], self.sites[name_next]
        site.index_t_check_next = site_prev.index_t_check_next
        site_prev.next_nb[:site.index_t_check_next + 1] = [site] * (site.index_t_check_next + 1)
        site.next_nb[:site.index_t_check_next + 1] = [site_next] * (site.index_t_check_next + 1)

    def binary_resample(self, site_prev: Site, site_next: Site, index: int):
        if site_prev.in_obs_at_index(index) == site_next.in_obs_at_index(index) and \
                not site_prev.interpolated and not site_next.interpolated and not self.mode_resamp_interp:
            return self.site_mngr.site_from_mean_init_cond(site_prev, site_next, index)
        else:
            return self.site_mngr.site_from_interpolation(site_prev, site_next, index)

    def check_solutions(self):
        to_check = set(self.sites.values()).difference(self._checked_for_solution).difference(
            set(site for site in self.sites.values() if len(site.ode_legs) == 0))
        for site in to_check:
            if np.any(np.sum(np.square(site.states[np.logical_not(np.logical_or(np.isnan(site.states[:, 0]),
                                                                                np.isnan(site.states[:,
                                                                                         1])))] - self.pb.x_target),
                             axis=-1)
                      < self._target_radius_sq):
                self.success = True
                self.solution_sites.add(site)
        min_cost = None
        solutions_sites_cost = {}
        for site in self.solution_sites:
            candidate_cost = np.min(
                site.costs[np.sum(np.square(site.states - self.pb.x_target), axis=-1) < self._target_radius_sq])
            solutions_sites_cost[site] = candidate_cost
            if min_cost is None or candidate_cost < min_cost:
                min_cost = candidate_cost
                self.solution_site = site
                self.solution_site_min_cost_index = list(self.solution_site.costs).index(min_cost)

        for site, cost in solutions_sites_cost.items():
            if cost > 1.15 * min_cost:
                self.solution_sites.remove(site)

        self._checked_for_solution.union(to_check)

    @property
    def found_guaranteed_solution(self):
        if self.solution_site_min_cost_index is None:
            return False
        return self.validity_index >= self.solution_site_min_cost_index

    def save_results(self, scale_length: Optional[float] = None, scale_time: Optional[float] = None,
                     bl: Optional[ndarray] = None, time_offset: Optional[float] = None):
        super(SolverEFResampling, self).save_results(scale_length, scale_time, bl, time_offset)
        if self.solution_site is not None:
            self.pb.io.save_traj(self.solution_site.traj, name='solution',
                                 scale_length=scale_length, scale_time=scale_time, bl=bl, time_offset=time_offset)

    def post_solve(self):
        super().post_solve()
        # with Chrono('Discretizing'):
        #     cache_count = 0
        #     sites_non_void = self.sites_non_void
        #     for site in tqdm(sites_non_void, file=sys.stdout):
        #         cache_count += site.discretize()
        #     total = len(sites_non_void) * self.n_time
        #     print(f'Cached: {cache_count}/{total} ({100 * cache_count/total:.2f}%)')


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
        self.stop = False

    def setup(self):
        super().setup()
        initial_sites = self.create_initial_sites()
        self.sites_valid[0] |= set(initial_sites)
        self._sites_created_at_subframe[0] |= set(initial_sites)
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
        count = 1
        dki = DelayedKeyboardInterrupt()
        with dki:
            while len(self._to_shoot_sites) > 0:
                if dki.signal_received:
                    self.stop = True
                    break
                print(f'Subloop {count:0>2}, VI: {self.validity_index}')
                while len(self._to_shoot_sites) > 0:
                    site = self._to_shoot_sites.pop()
                    self.integrate_site_to_target_time(site, self.times[self.index_t_next_subframe - 1])
                new_sites = self.feed_new_sites(index_t_hi=self.index_t_next_subframe - 1)
                self._to_shoot_sites.extend(new_sites)
                self._sites_created_at_subframe[self.i_subframe] |= set(new_sites)
                count += 1
        print(f'End         VI: {self.validity_index}')
        if not self.stop:
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
            dist = np.linalg.norm(site.state_cur - self.pb.x_target) if self.pb.coords == Coords.CARTESIAN else \
                Utils.geodesic_radian_distance(site.state_cur, self.pb.x_target)
            sup_time = dist / (self.pb.srf_max + self._ff_max_norm)
            if site.t_cur + sup_time > self.times[-1]:
                site.close(ClosureReason.SUBOPTI_DISTANCE)
                continue
            value_opti = Utils.interpolate(self._cost_map.values, bl, spacings,
                                           site.state_at_index(self.index_t_next_subframe - 1))
            if np.isnan(value_opti):
                new_valid_sites.append(site)
                continue
            if site.cost_at_index(self.index_t_next_subframe - 1) > 1.02 * value_opti:
                site.close(ClosureReason.SUBOPTIMAL)
            else:
                new_valid_sites.append(site)
        self.sites_valid[self.i_subframe + 1] |= set(new_valid_sites)

    def solve(self):
        self.setup()
        self.pre_solve()
        for _ in range(self.n_subframes):
            if self.stop:
                break
            s = 'Subframe {i_subframe}/{n_subframe}, Act: {n_active: >4}, Cls: {n_closed: >4} (Tot: {n_total: >4})'
            print(s.format(
                i_subframe=self.i_subframe + 1,
                n_subframe=self.n_subframes,
                n_active=len(self.sites_valid[self.i_subframe]),
                n_closed=len([site for site in self.sites.values() if site.closed]),
                n_total=len(self.sites))
            )
            self.step()
        self.check_solutions()
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
            for index_t in range(max(0, index_t_lo), min(index_t_hi + 1, site.index_t_check_next)):
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


def leq_float(a, b):
    return a <= b or np.isclose(a, b)


def ls_float(a, b):
    return a < b and not np.isclose(a, b)


def geq_float(a, b):
    return leq_float(-a, -b)


def gs_float(a, b):
    return ls_float(-a, -b)
