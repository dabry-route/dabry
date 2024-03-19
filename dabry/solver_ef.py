import json
import math
import os
import signal
import sys
import warnings
from abc import ABC
from enum import Enum
from typing import Optional, Dict, Iterable, List

import numpy as np
from numpy import ndarray
from scipy.integrate import solve_ivp, OdeSolution
from scipy.integrate._ivp.ivp import OdeResult
from tqdm import tqdm

from dabry.misc import directional_timeopt_control, Utils, triangle_mask_and_cost, non_terminal, to_alpha, \
    diadic_valuation, alpha_to_int, Chrono
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
                 obs_name: Optional[str] = None, obs_trigo: Optional[bool] = None):
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
        self.events_dict: Dict[str, ndarray] = events_dict
        self.obs_name: str = obs_name
        self.obs_trigo: bool = obs_trigo


class IntStatus(Enum):
    FREE = 0
    OBSTACLE = 1
    OBS_ENTRY = 2
    OBS_EXIT = 3
    CLOSED = 4
    END = 5


class ClosureReason(Enum):
    SUBOPTIMAL = 0
    IMPOSSIBLE_OBS_TRACKING = 1


class NeuteringReason(Enum):
    # When two neighbours enter obstacle and have different directions
    OBSTACLE_SPLITTING = 0
    # Two neighbors experience a different obstacle history: resampling undefined yet
    DIFFERENT_OBS_HISTORY = 1


class Site:

    def __init__(self, t_init: float, cost_init: float, state_init: ndarray, costate_init: ndarray,
                 tau_exit: ndarray, time_grid: ndarray, name: str):
        self.t_init = t_init
        self.ode_legs: List[OdeAugResult] = []
        self.status_int: IntStatus = IntStatus.FREE

        self.cost_init = cost_init
        # TODO: are hard copies really required here ?
        self.state_init = state_init.copy()
        self.costate_init = costate_init.copy()
        self.tau_exit = tau_exit.copy()

        self.traj: Optional[Trajectory] = None
        self.index_t_check_next = 0
        self.prev_index_t_check_next = 0
        self.closure_reason: Optional[ClosureReason] = None
        self.neutering_reason: Optional[NeuteringReason] = None
        self.t_closed: Optional[float] = None
        self.t_neutered: Optional[float] = None
        self.time_grid: ndarray = time_grid
        self.next_nb: list[Optional[Site]] = [None] * time_grid.size
        self.name = name

    def discretize(self):
        states = np.nan * np.ones((self.n_time, 2))
        costates = np.nan * np.ones((self.n_time, 2))
        cost = np.nan * np.ones(self.n_time)
        for it, t in enumerate(self.time_grid):
            if not self.ensure_time_bounds(t):
                continue
            else:
                cost[it] = self.cost(t)
                states[it, :] = self.state(t)
                costates[it, :] = self.costate(t)
        self.traj = Trajectory(self.time_grid, states, costates=costates, cost=cost)

    def time_to_index(self, t: float):
        return np.searchsorted(self.time_grid, t, side='right') - 1

    @property
    def index_neutered(self):
        if self.t_neutered is None:
            return -1
        return self.time_to_index(self.t_neutered)

    def index_closed(self):
        if self.t_closed is None:
            return None
        return self.time_to_index(self.t_closed)

    @property
    def t_cur(self):
        if len(self.ode_legs) == 0:
            return self.t_init
        else:
            return self.ode_legs[-1].sol.t_max

    @property
    def t_exit_next(self):
        i_tau_exit = self.count_obs_cur
        if i_tau_exit >= self.tau_exit.size:
            return None
        else:
            return self.t_cur + self.tau_exit[i_tau_exit]

    @property
    def depth(self):
        if '-' not in self.name:
            return 0
        _, depth, _ = self.name.split('-')
        return int(depth)

    @property
    def closed(self):
        return self.closure_reason is not None

    def close(self, t: float, reason: ClosureReason):
        self.closure_reason = reason
        self.t_closed = t

    @property
    def neutered(self):
        return self.neutering_reason is not None

    def neuter(self, t: float, reason: NeuteringReason):
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

    def ensure_time_bounds(self, t: float):
        if len(self.ode_legs) == 0:
            return False
        if t < self.ode_legs[0].sol.t_min or t > self.ode_legs[-1].sol.t_max:
            return False
        return True

    def obs_at(self, t: float):
        if len(self.ode_legs) == 0:
            return None
        ode_leg = self.find_leg(t)
        return ode_leg.obs_name

    def obs_at_index(self, index: int):
        return self.obs_at(self.time_grid[index])

    def in_obs_at_index(self, index: int):
        return self.obs_at_index(index) is not None

    def obs_history(self):
        if len(self.ode_legs) == 0:
            return []
        res = []
        for ode_leg in self.ode_legs:
            if ode_leg.obs_name is not None:
                res = res.append(ode_leg.obs_name)
        return res

    @property
    def leg_prev_obs(self):
        if len(self.ode_legs) == 0:
            raise ValueError('No legs')
        for ode_leg in self.ode_legs[::-1]:
            if ode_leg.obs_name is not None:
                return ode_leg
        raise ValueError('No obstacle leg')

    def trigo_at(self, t: float):
        if len(self.ode_legs) == 0:
            return None
        ode_leg = self.find_leg(t)
        return ode_leg.obs_trigo

    def trigo_at_index(self, index: int):
        return self.trigo_at(self.time_grid[index])

    def captive(self):
        if len(self.ode_legs) == 0:
            return False
        last_leg = self.ode_legs[-1]
        if last_leg.obs_name is None:
            # Last leg is a free leg, see if it entered an obstacle
            active_obstacles = [name for name, t_events in last_leg.events_dict.items() if t_events.shape[0] > 0]
            if len(active_obstacles) >= 2:
                warnings.warn("Multiple active obstacles", category=RuntimeWarning)
            if len(active_obstacles) == 1:
                return True
            return False
        else:
            # Last leg is a captive leg, see if it exits
            exit_events = [name for name, t_events in last_leg.events_dict.items() if t_events.shape[0] > 0]
            if len(exit_events) > 0:
                return False
            return True

    @property
    def obs_cur(self):
        if len(self.ode_legs) == 0:
            return None
        return self.ode_legs[-1].obs_name

    @property
    def count_obs_cur(self):
        if len(self.ode_legs) == 0:
            return 0
        return sum([1 if ode_leg.obs_name is not None else 0 for ode_leg in self.ode_legs])

    @property
    def trigo_cur(self):
        if len(self.ode_legs) == 0:
            return None
        return self.ode_legs[-1].obs_trigo

    def time_at_index(self, index: int):
        return self.time_grid[index]

    def cost_at_index(self, index: int):
        return self.cost(self.time_grid[index])

    def state_at_index(self, index: int):
        return self.state(self.time_grid[index])

    def costate_at_index(self, index: int):
        return self.costate(self.time_grid[index])

    def find_leg(self, t: float):
        if not self.ensure_time_bounds(t):
            raise ValueError(f'Time out of bounds ({t})')
        if len(self.ode_legs) == 0:
            raise ValueError('Empty leg list')
        for ode_leg in self.ode_legs:
            if ode_leg.sol.t_min <= t <= ode_leg.sol.t_max:
                return ode_leg
        raise ValueError(f'Time not in ODE legs ({t})')

    def cost(self, t: float):
        return self.find_leg(t).sol(t)[0]

    def state(self, t: float):
        return self.find_leg(t).sol(t)[1:3]

    def costate(self, t: float):
        sol_t = self.find_leg(t).sol(t)
        if sol_t.size <= 3:
            return np.nan * np.ones(2)
        return sol_t[3:5]

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
        self.ode_legs.append(ode_aug_res)
        self.update_status()

    def update_status(self):
        # costate_cur
        # int_status
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

    def site_from_parents(self, site_prev, site_next, costate_init: Optional[ndarray] = None,
                          tau_exit: Optional[ndarray] = None) -> Site:
        name = self.name_from_parents_name(site_prev.name, site_next.name)
        tau_exit = tau_exit if tau_exit is not None else \
            0.5 * (site_prev.tau_exit + site_next.tau_exit)
        costate_init = costate_init if costate_init is not None else \
            0.5 * (site_prev.costate_init + site_next.costate_init)
        site_child = Site(site_prev.t_init,
                          0.5 * (site_prev.cost_init + site_next.cost_init),
                          0.5 * (site_prev.state_init + site_next.state_init),
                          costate_init,
                          tau_exit, self.time_grid, name)
        return site_child


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
        self.traj_groups = []

    @property
    def trajs(self) -> list[Trajectory]:
        return []

    @property
    def t_upper_bound(self):
        return self.t_init + self.total_duration

    def dyn_augsys(self, t: float, y: ndarray):
        return np.hstack((1., self.pb.augsys_dyn_timeopt(t, y[1:3], y[3:5])))

    def dyn_constr(self, t: float, x: ndarray, obstacle: str, trigo: bool):
        sign = 2 * trigo - 1
        d = np.array(((0., -sign), (sign, 0.))) @ self.obstacles[obstacle].d_value(x[1:3])
        ff_val = self.pb.model.ff.value(t, x[1:3])
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
            res: OdeResult = solve_ivp(self.dyn_augsys, (self.t_init, self.t_upper_bound),
                                       np.array(tuple(self.pb.x_init) + tuple(costate)), t_eval=t_eval,
                                       events=list(self.events.values()), **self._integrator_kwargs)
            traj = Trajectory(res.t, res.y.transpose()[:, :2], costates=res.y.transpose()[:, 2:],
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
        self.initial_sites: list[Site] = []
        self._to_shoot_sites: list[Site] = []
        self._checked_for_solution: set[Site] = set()
        self.solution_sites: set[Site] = set()
        self.solution_site: Optional[Site] = None
        self.solution_site_min_cost_index: Optional[int] = None
        self.mode_origin: bool = mode_origin
        self.site_mngr: SiteManager = SiteManager(self.n_costate_sectors, self.max_depth, self.times)
        self._ff_max_norm = np.max(np.linalg.norm(self.pb.model.ff.values, axis=-1)) if \
            hasattr(self.pb.model.ff, 'values') else None

    def create_initial_sites(self):
        self.initial_sites = [
            Site(self.t_init, 0, self.pb.x_init, np.array((np.cos(theta), np.sin(theta))), np.array(()), self.times,
                 self.site_mngr.name_from_pdl(i_theta, 0, 0))
            for i_theta, theta in enumerate(np.linspace(0, 2 * np.pi, self.n_costate_sectors, endpoint=False))
        ]
        self.sites = {site.name: site for site in self.initial_sites}
        return self.initial_sites

    def connect_initial_sites(self):
        for i_site, site in enumerate(self.initial_sites):
            site.next_nb[0] = self.initial_sites[(i_site + 1) % len(self.initial_sites)]

    @property
    def trajs(self):
        return list(map(lambda x: x.traj, [site for site in self.sites.values() if site.traj is not None]))

    def integrate_site_to_target_time(self, site: Site, t_target: float):
        if site.status_int == IntStatus.CLOSED:
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
                if not np.isclose(dot, 0):
                    # close site
                    break
                x0 = np.hstack((site.cost_cur, state_cur))
                ode_aug_res = self.integrate_captive(site.t_cur, t_target, x0, obs_name, trigo, site.t_exit_next)
                site.add_leg(ode_aug_res)
            elif site.status_int == IntStatus.OBSTACLE:
                x0 = np.hstack((site.cost_cur, site.state_cur))
                ode_aug_res = self.integrate_captive(site.t_cur, t_target, x0, site.obs_cur, site.trigo_cur,
                                                     site.t_exit_next)
                site.add_leg(ode_aug_res)
            elif site.status_int == IntStatus.OBS_EXIT:
                control_cur = self.dyn_constr(site.t_cur, np.hstack((site.cost_cur, site.state_cur)),
                                              site.obs_cur, site.trigo_cur)[1:3] - \
                              self.pb.model.ff.value(site.t_cur, site.state_cur)
                # TODO: order 2 detection of obstacle penetration
                costate_artificial = - control_cur / np.linalg.norm(control_cur)
                y0 = np.hstack((site.cost_cur, site.state_cur, costate_artificial))
                ode_aug_res = self.integrate_free(site.t_cur, t_target, y0)
                site.add_leg(ode_aug_res)

    def integrate_free(self, t_start: float, t_end: float, y0: ndarray):
        res: OdeResult = solve_ivp(self.dyn_augsys, (t_start, t_end), y0,
                                   events=list(self.events.values()), dense_output=True, **self._integrator_kwargs)
        events_dict = self.t_events_to_dict(res.t_events)
        return OdeAugResult(res, events_dict, None)

    def integrate_captive(self, t_start: float, t_end: float, x0: ndarray, obs_name: str, trigo: bool,
                          t_exit: Optional[float] = None):
        t_hi = min(t_exit, t_end) if t_exit is not None else t_end
        res: OdeResult = solve_ivp(self.dyn_constr, (t_start, t_hi),
                                   x0, events=[self._event_exit_obs], args=[obs_name, trigo],
                                   dense_output=True, **self._integrator_kwargs)
        events_dict = {
            "exit_forced": res.t_events[0],
            "exit_manual": np.array(t_exit) if res.t_events[0].size == 0 and not np.isclose(res.sol.t_max, t_end) else
            np.array(())
        }
        return OdeAugResult(res, events_dict, obs_name, trigo)

    @property
    def validity_list(self):
        return \
            [
                (site.index_t_check_next, site) for site in self.sites_non_void()
                if not (site.neutered and site.index_t_check_next + 1 >= site.index_neutered) and
                   not (site.closed and site.index_t_check_next == site.index_t) and
                   not (site.next_nb[site.index_t_check_next].closed and site.index_t_check_next + 1 >=
                        site.next_nb[site.index_t_check_next].index_closed)
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

    def trim_distance(self):
        for site in self.sites.values():
            sup_time = np.linalg.norm(site.state_at_index(site.index_t) - self.pb.x_target) / (self.pb.srf_max +
                                                                                               self._ff_max_norm)
            if self.times[-1] - self.times[site.index_t] > sup_time:
                site.close(ClosureReason.SUBOPTIMAL)

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
        # TODO: experimental
        for site in self.sites_non_void():
            site.discretize()
        self.post_solve()

    def sites_non_void(self):
        return [site for site in self.sites.values() if len(site.ode_legs) > 0]

    def get_cost_map(self):
        self._cost_map.update_from_sites(self.sites_non_void(), 0, self.validity_index)
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
            self.connect_initial_sites()
            return new_sites
        else:
            new_sites = self.create_new_sites(index_t_hi)
            self.connect_sites(new_sites)
            return new_sites

    def create_new_sites(self, index_t_hi: Optional[int] = None) -> list[Site]:
        index_t_hi = index_t_hi if index_t_hi is not None else self.n_time - 1
        prev_sites = [site for site in self.sites.values()]
        new_sites: list[Site] = []
        for i_s, site in enumerate(prev_sites):
            site_nb = site.next_nb[site.index_t_check_next]
            new_id_check_next = site.index_t_check_next
            new_site: Optional[Site] = None
            upper_index = min(site.index_t + 1, site_nb.index_t + 1, index_t_hi)
            for i in range(site.index_t_check_next, upper_index):
                state = site.state_at_index(i)
                state_nb = site_nb.state_at_index(i)
                if site.in_obs_at_index(i) and site_nb.in_obs_at_index(i) and \
                        site.obs_at_index(i) == site_nb.obs_at_index(i) and \
                        (site.trigo_at_index(i) != site_nb.trigo_at_index(i)):
                    site.neuter(i, NeuteringReason.OBSTACLE_SPLITTING)
                    break
                if np.sum(np.square(state - state_nb)) > self._max_dist_sq and \
                        site.depth < self.max_depth - 1 and site_nb.depth < self.max_depth - 1 and not site.neutered:
                    # Resampling
                    # if site.obs_history() != site_nb.obs_history():
                    #     # TODO: experimental
                    #     # site.neuter(i, NeuteringReason.DIFFERENT_OBS_HISTORY)
                    #     # break
                    #     pass
                    new_site = self.binary_resample(site, site_nb, i)
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

    def connect_sites(self, sites: list[Site]):
        for site in sites:
            name_prev, name_next = self.site_mngr.parents_name_from_name(site.name)
            site_prev, site_next = self.sites[name_prev], self.sites[name_next]
            index_t_check_next = site_prev.index_t_check_next
            site_prev.next_nb[:index_t_check_next + 1] = [site] * (index_t_check_next + 1)
            site.next_nb[:index_t_check_next + 1] = [site_next] * (index_t_check_next + 1)

    def binary_resample(self, site_prev: Site, site_next: Site, index: int):
        if site_prev.in_obs_at_index(index) == site_next.in_obs_at_index(index):
            # Either free and free or obs and obs
            # Assume same switching structure
            # (shall be asymptotically true when the distance tolerance between extremals goes to zero)
            cond = np.array_equal(site_prev.tau_exit, site_next.tau_exit)
            # assert cond
            return self.site_mngr.site_from_parents(site_prev, site_next)
        else:
            site_in_obs, site_out_obs = (site_prev, site_next) if site_prev.in_obs_at_index(index) else \
                (site_next, site_prev)
            # Assume switching structure differs only by one between neighbours
            # (shall be true asymptotically)
            cond = abs(site_out_obs.tau_exit.size - site_in_obs.tau_exit.size) <= 1
            assert cond
            if site_out_obs.tau_exit.size == 0:
                leg_prev_obs = site_in_obs.leg_prev_obs
                if site_in_obs.tau_exit.size > 0:
                    duration_obs = site_in_obs.tau_exit[-1]
                else:
                    duration_obs = (leg_prev_obs.sol.t_max - leg_prev_obs.sol.t_min)
                tau_exit = np.array((duration_obs / 2,))
            else:
                tau_exit = site_out_obs.tau_exit.copy()
                leg_prev_obs = site_in_obs.leg_prev_obs
                duration_obs = (leg_prev_obs.sol.t_max - leg_prev_obs.sol.t_min)
                tau_exit[-1] = (tau_exit[-1] + duration_obs) / 2
            return self.site_mngr.site_from_parents(site_prev, site_next, costate_init=site_in_obs.costate_init,
                                                    tau_exit=tau_exit)

    def check_solutions(self):
        to_check = set(self.sites.values()).difference(self._checked_for_solution).difference(
            set(site for site in self.sites.values() if site.traj is None))
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

    def save_results(self, scale_length: Optional[float] = None, scale_time: Optional[float] = None,
                     bl: Optional[ndarray] = None, time_offset: Optional[float] = None):
        super(SolverEFResampling, self).save_results(scale_length, scale_time, bl, time_offset)
        if self.solution_site is not None:
            if self.solution_site.traj is None:
                self.solution_site.discretize()
            self.pb.io.save_traj(self.solution_site.traj, name='solution',
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
            print(self.validity_index)
            while len(self._to_shoot_sites) > 0:
                site = self._to_shoot_sites.pop()
                self.integrate_site_to_target_index(site, self.index_t_next_subframe - 1)
                if site.traj is not None:
                    if not site.is_root() and not site.has_neighbours:
                        self.connect_to_parents(site)
                cond = site.index_t == self.index_t_next_subframe - 1 or site.closed
                assert cond
            new_sites = self.feed_new_sites(index_t_hi=self.index_t_next_subframe - 1)
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
            for index_t in range(index_t_lo, min(index_t_hi + 1, site.index_t_check_next)):
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
