import json
import math
import os
import warnings
from datetime import datetime
from typing import Optional, List

import numpy as np
import scipy.integrate as scitg
from numpy import ndarray

from dabry.aero import MermozAero, Aero
from dabry.feedback import GSTargetFB, Feedback, HTargetFB
from dabry.flowfield import RankineVortexFF, UniformFF, DiscreteFF, StateLinearFF, RadialGaussFF, GyreFF, \
    PointSymFF, LinearFFT, BandFF, TrapFF, ChertovskihFF, \
    FlowField, VortexFF, ZeroFF, WrapperFF, GyreMSEASFF
from dabry.io_manager import IOManager
from dabry.misc import Utils, csv_to_dict, Coords
from dabry.model import Model
from dabry.obstacle import CircleObs, FrameObs, Obstacle, WrapperObs
from dabry.penalty import Penalty, NullPenalty, WrapperPen
from dabry.trajectory import Trajectory

"""
problem.py
Handles navigation problem data.

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


class NavigationProblem:
    ALL = csv_to_dict(os.path.join(os.path.dirname(__file__), 'problems.csv'))

    def __init__(self, ff: FlowField, x_init: ndarray, x_target: ndarray, srf_max: float,
                 obstacles: Optional[List[Obstacle]] = None,
                 bl: Optional[ndarray] = None,
                 tr: Optional[ndarray] = None,
                 name: Optional[str] = None,
                 target_radius: Optional[float] = None,
                 aero: Optional[Aero] = None,
                 penalty: Optional[Penalty] = None,
                 autoframe=True):
        self.model = Model.zermelo(ff)
        self.x_init: ndarray = x_init.copy()
        self.x_target: ndarray = x_target.copy()
        self.srf_max: float = srf_max
        self.aero: Aero = MermozAero() if aero is None else aero

        # Domain bounding box corners
        self.bl: ndarray = bl.copy() if bl is not None else np.array(())
        self.tr: ndarray = tr.copy() if tr is not None else np.array(())

        self.length_reference = np.min(tr - bl) if bl is not None and tr is not None else \
            np.linalg.norm(x_target - x_init)

        self.name = 'Unnamed problem' if name is None else name

        self.obstacles: list[Obstacle] = obstacles if obstacles is not None else []
        self.penalty: Penalty = NullPenalty() if penalty is None else penalty

        self.io = IOManager(self.name)

        # Bound computation domain on wind grid limits
        if self.bl.size == 0 or self.tr.size == 0:
            ff = self.model.ff
            if hasattr(ff, 'bounds'):
                self.bl = np.array((ff.bounds[-2, 0], ff.bounds[-1, 0]))
                self.tr = np.array((ff.bounds[-2, 1], ff.bounds[-1, 1]))
                x_init_out_of_bounds = np.any(self.x_init < self.bl) or np.any(self.x_init > self.tr)
                x_target_out_of_bounds = np.any(self.x_target < self.bl) or np.any(self.x_target > self.tr)

                if x_init_out_of_bounds or x_target_out_of_bounds:
                    warnings.warn('When setting problem bounds to ff bounds, the following were out of bounds: '
                                  f'{"x_init " if x_init_out_of_bounds else ""}'
                                  f'{"x_target" if x_target_out_of_bounds else ""}', category=UserWarning)
            else:
                w = 1.15 * self.length_reference
                self.bl = (self.x_init + self.x_target) / 2. - np.array((w / 2., w / 2.))
                self.tr = (self.x_init + self.x_target) / 2. + np.array((w / 2., w / 2.))

        self.target_radius: float = target_radius if target_radius is not None else \
            0.025 * np.linalg.norm(self.tr - self.bl)

        frame_offset = 0.
        if autoframe:
            bl_frame = self.bl + (self.tr - self.bl) * frame_offset / 2.
            tr_frame = self.tr - (self.tr - self.bl) * frame_offset / 2.
            self.obs_frame = FrameObs(bl_frame, tr_frame)
            self.obstacles.append(self.obs_frame)
        # self.obstacles.extend(NavigationProblem.spherical_frame(bl, tr, offset_rel=0))

    def get_grid_params(self, nx: int, ny: int) -> tuple[ndarray, ndarray]:
        """
        Get the (bottom-left corner, spacings) representation of the underlying grid
        for the required discretization
        :param nx: Discretization number on first axis
        :param ny: Discretization number on second axis
        :return: (bottom-left corner, spacings)
        """
        return self.bl, (self.tr - self.bl) / (np.array((nx, ny)) - np.ones(2).astype(np.int32))

    def update_ff(self, ff: FlowField):
        self.model = Model.zermelo(ff)

    def save_ff(self):
        self.io.save_ff(self.model.ff, bl=self.bl, tr=self.tr)

    def save_info(self):
        pb_info = {
            'x_init': tuple(self.x_init),
            'x_target': tuple(self.x_target),
            'srf_max': self.srf_max,
            'target_radius': self.target_radius,
            'bl': self.bl.tolist(),
            'tr': self.tr.tolist(),
            'time_orthodromic': self.time_orthodromic(),
            'time_htarget': self.time_htarget()
        }
        with open(os.path.join(self.io.case_dir, f'{self.name}.json'), 'w') as f:
            json.dump(pb_info, f, indent=4)

    @property
    def coords(self):
        return self.model.coords

    def __str__(self):
        return str(self.name)

    def dualize(self):
        """
        :return: The dual problem, i.e. going from target to init in the mirror wind (multiplication by -1)
        """
        kwargs = {}
        for k, v in self.__dict__.items():
            kwargs[k] = v
        kwargs['ff'] = self.model.ff.dualize()
        kwargs['x_init'] = self.x_target.copy()
        kwargs['x_target'] = self.x_init.copy()

        for attr in ['model', 'geod_l', 'name', 'io']:
            del kwargs[attr]

        return NavigationProblem(**kwargs)

    def distance(self, x1, x2):
        return Utils.distance(x1, x2, self.coords)

    def middle(self, x1, x2):
        return Utils.middle(x1, x2, self.coords)

    def timeopt_control_cartesian(self, costate: ndarray):
        return timeopt_control_cartesian(costate, self.srf_max)

    def timeopt_control_gcs(self, state: ndarray, costate: ndarray):
        return timeopt_control_gcs(state, costate, self.srf_max)

    def augsys_dyn_timeopt(self, t: float, state: ndarray, costate: ndarray, control: ndarray):
        return np.hstack((self.model.dyn.value(t, state, control),
                          -self.model.dyn.d_value__d_state(t, state, control).transpose() @ costate
                          - self.penalty.d_value(t, state)))

    def augsys_dyn_timeopt_cartesian(self, t: float, state: ndarray, costate: ndarray):
        return self.augsys_dyn_timeopt(t, state, costate, self.timeopt_control_cartesian(costate))

    def augsys_dyn_timeopt_gcs(self, t: float, state: ndarray, costate: ndarray):
        return self.augsys_dyn_timeopt(t, state, costate, self.timeopt_control_gcs(state, costate))

    def hamiltonian(self, t: float, state: ndarray, costate: ndarray, control: ndarray):
        return costate @ (control + self.model.ff.value(t, state)) + 1

    def hamiltonian_reduced(self, t: float, state: ndarray, costate: ndarray):
        # TODO : adapt to gcs
        return self.hamiltonian(t, state, costate, self.timeopt_control_cartesian(costate))

    def in_obs(self, state):
        return [obs for obs in self.obstacles if obs.value(state) < 0.]

    def apply_feedback(self, fb: Feedback, rel_timeout=10, n_time=1000) -> Trajectory:
        """
        Integrate a trajectory applying feedback control
        :param fb: The feedback control law
        :param rel_timeout: The time scale multiplicator to build the timeout date
        :param n_time: The time discretization number
        :return: A Trajectory
        """
        def dyn_fb(t, x):
            return self.model.dyn(t, x, fb(t, x))

        time_scale = self.length_reference / self.srf_max
        t_max = rel_timeout * time_scale
        times = np.linspace(self.model.ff.t_start, self.model.ff.t_start + t_max, n_time)
        _target_radius_sq = self.target_radius ** 2

        def event_target(t, x):
            return np.sum(np.square(x - self.x_target)) - _target_radius_sq

        event_target.terminal = True
        res = scitg.solve_ivp(dyn_fb, (times[0], times[-1]), self.x_init, t_eval=times, events=[event_target])
        return Trajectory(res.t, res.y.transpose(), self.model.coords, events={'target': res.t_events[0]})

    def auto_time_upper_bound(self):
        return min(self.time_orthodromic(), self.time_htarget())

    def orthodromic(self):
        """
        Compute the orthodromic trajectory between start and target
        :return: Trajectory
        """
        fb = GSTargetFB(self.model.ff, self.srf_max, self.x_target)
        return self.apply_feedback(fb)

    def htarget(self):
        fb = HTargetFB(self.x_target, self.model.coords)
        return self.apply_feedback(fb)

    def time_orthodromic(self):
        traj_ortho = self.orthodromic()
        reached = traj_ortho.events['target'].size > 0
        return traj_ortho.events['target'][0] if reached else np.infty

    def time_htarget(self):
        traj_htarget = self.htarget()
        reached = traj_htarget.events['target'].size > 0
        return traj_htarget.events['target'][0] if reached else np.infty

    def scaling_params(self):
        if self.coords == Coords.CARTESIAN:
            scale_length = self.length_reference
            x_init = (self.x_init - self.bl) / scale_length
            x_target = (self.x_target - self.bl) / scale_length
            bl_pb_adim = np.zeros(2)
            bl_wrapper = self.bl.copy()
            tr_pb_adim = (self.tr - self.bl) / scale_length
            srf_max = self.srf_max
        else:
            # Case self.coords == Coords.GCS
            # Do not rescale lon/lat because computation will be false
            # TODO: think about shifting longitude to zero to avoid periodic bounds
            scale_length = 1.
            x_init = self.x_init
            x_target = self.x_target
            bl_pb_adim = self.bl
            bl_wrapper = np.zeros(2)
            tr_pb_adim = self.tr
            # In GCS convention, srf is specified in meters per seconds and
            # has to be cast to radians per seconds before regular scaling
            srf_max = self.srf_max / Utils.EARTH_RADIUS

        scale_time = scale_length / srf_max
        return x_init, x_target, bl_pb_adim, bl_wrapper, tr_pb_adim, scale_length, scale_time

    def rescale(self):
        """
        Builds a new problem where space and time variables are of unit magnitude
        and srf_max is 1
        :return: A rescaled NavigationProblem
        """
        x_init, x_target, bl_pb_adim, bl_wrapper, tr_pb_adim, scale_length, scale_time = self.scaling_params()
        # In the new system, srf_max is unit
        srf_max = 1.

        wrapper_ff = WrapperFF(self.model.ff, scale_length, bl_wrapper, scale_time, self.model.ff.t_start)

        obstacles = self.obstacles.copy()
        obstacles.remove(self.obs_frame)
        obstacles = [WrapperObs(obs, scale_length, bl_wrapper) for obs in obstacles]
        penalty = WrapperPen(self.penalty, scale_length, bl_wrapper, scale_time, self.model.ff.t_start)
        return NavigationProblem(wrapper_ff, x_init, x_target, srf_max,
                                 bl=bl_pb_adim, tr=tr_pb_adim, obstacles=obstacles, penalty=penalty,
                                 name=self.name + ' (scaled)')

    @classmethod
    def from_database(cls, x_init: ndarray, x_target: ndarray, srf: float,
                      t_start: [float, datetime], t_end: [float, datetime],
                      resolution='0.5', pressure_level='1000', data_path: Optional[str] = None):
        """
        :param x_init: Initial position (lon, lat)
        :param x_target: Target position (lon, lat)
        :param srf: Speed relative to flow
        :param t_start: Time window lower bound
        :param t_end: Time window upper bound
        :param resolution: The weather model grid resolution in degrees, e.g. '0.5'
        :param pressure_level: The pressure level in hPa, e.g. '1000', '500', '200'
        :param data_path: Force path to data
        :return:
        """
        bl_lon = Utils.RAD_TO_DEG * min(Utils.ang_principal(x_init[0]), Utils.ang_principal(x_target[0]))
        bl_lat = Utils.RAD_TO_DEG * min(x_init[1], x_target[1])
        bl_lon = math.floor((bl_lon - 5) / 10) * 10
        bl_lat = math.floor((bl_lat - 5) / 10) * 10
        bl = Utils.DEG_TO_RAD * np.array((bl_lon, bl_lat))
        tr_lon = Utils.RAD_TO_DEG * max(Utils.ang_principal(x_init[0]), Utils.ang_principal(x_target[0]))
        tr_lat = Utils.RAD_TO_DEG * max(x_init[1], x_target[1])
        tr_lon = math.ceil((tr_lon + 5) / 10) * 10
        tr_lat = math.ceil((tr_lat + 5) / 10) * 10
        tr = Utils.DEG_TO_RAD * np.array((tr_lon, tr_lat))
        grid_bounds = np.array((bl, tr)).transpose()
        ff = DiscreteFF.from_cds(grid_bounds, t_start, t_end, resolution=resolution,
                                 pressure_level=pressure_level, data_path=data_path)
        start_date = t_start if isinstance(t_start, float) else t_start.timestamp()
        case_name = IOManager.format_cname(x_init, x_target, start_date)
        return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=case_name)

    @classmethod
    def base_name(cls, name: str):
        for b_name, attrs in cls.ALL.items():
            if b_name == name:
                return name
            if attrs['s_name'] == name:
                return b_name
        raise ValueError('Cannot find problem "%s". Check "problems.csv"' % name)

    @classmethod
    def from_name(cls, name: str):
        """
        Creates a NavigationProblem from problem list
        :param name: Full name as a string. May be the short name of a problem as defined in ALL.
        :return: Corresponding NavigationProblem
        """
        b_name = cls.base_name(name)
        if b_name == "linear":
            srf = 1.

            f = 1.
            x_init = f * np.array([0., 0.])
            x_target = f * np.array([1., 0.])

            g = 1.
            gradient = np.array([[0., g], [0., 0.]])
            origin = np.array([0., 0.])
            value_origin = np.array([0., 0.])

            bl = f * np.array([-0.2, -1.])
            tr = f * np.array([1.2, 1.])

            ff = StateLinearFF(gradient, origin, value_origin)

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "three_vortices":
            srf = 1.

            x_init = np.array([0., 0.])
            x_target = np.array([1., 0.])

            bl = np.array([-0.1, -1])
            tr = np.array([1.1, 1])

            ff = sum([VortexFF(np.array((0.5, 0.5)), 1.),
                      VortexFF(np.array((0.5, 0.)), -1),
                      VortexFF(np.array((0.5, -0.5)), 1)], ZeroFF())
            obs = [CircleObs(np.array((0.5, 0.5)), 0.1),
                   CircleObs(np.array((0.5, 0.)), 0.1),
                   CircleObs(np.array((0.5, -0.5)), 0.1)]

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, obstacles=obs, name=b_name)

        if b_name == "gyre":
            srf = 1

            sf = 1.

            x_init = sf * np.array((0.6, 0.6))
            x_target = sf * np.array((2.4, 2.4))
            # bl = sf * np.array((0.5, 0.5))
            # tr = sf * np.array((2.5, 2.5))

            ff = GyreFF(0.5, 0.5, 2., 2., 1)

            return cls(ff, x_init, x_target, srf, name=b_name)

        if b_name == "gyre_li2020":
            v_a = 0.6

            sf = 1.

            x_init = sf * np.array((125., 125.))
            x_target = sf * np.array((375., 375.))
            bl = sf * np.array((-10, -10))
            tr = sf * np.array((510, 510))

            ff = GyreFF(0., 0., 500., 500., 1.)

            return cls(ff, x_init, x_target, v_a, bl=bl, tr=tr, name=b_name)

        if b_name == "point_symmetric_techy2011":
            srf = 1.
            sf = 1.
            x_init = sf * np.array((0., 0.))
            x_target = sf * np.array((1., 0.))
            bl = sf * np.array((-0.1, -1.))
            tr = sf * np.array((1.1, 1.))

            # To get w as wind value at start point, choose gamma = w / 0.583
            gamma = srf / sf * 1.
            omega = 0.
            ff = PointSymFF(sf * np.array((0.5, 0.3)), gamma, omega)

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "three_obstacles":
            srf = 1.
            sf = 1.

            x_init = sf * np.array((0., 0.))
            x_target = sf * np.array((1., 0.))
            bl = sf * np.array((-0.15, -0.65))
            tr = sf * np.array((1.15, 0.65))

            const_wind = UniformFF(np.array([.1, .1]))

            c1 = sf * np.array((0.5, 0.))
            c2 = sf * np.array((0.5, 0.1))
            c3 = sf * np.array((0.5, -0.1))

            wind_obstacle1 = RadialGaussFF(c1, sf * 0.1, 0.5 * 0.2, srf * 5.)
            wind_obstacle2 = RadialGaussFF(c2, sf * 0.1, 0.5 * 0.2, srf * 5.)
            wind_obstacle3 = RadialGaussFF(c3, sf * 0.1, 0.5 * 0.2, srf * 5.)

            ff = wind_obstacle1 + wind_obstacle2 + wind_obstacle3 + const_wind

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "big_rankine":
            srf = 0.01

            x_init = np.array([0.2, 0.5])
            x_target = np.array([1.3, 0.5])

            bl = np.array((-0.3, -0.3))
            tr = np.array((1.6, 1.3))

            ff = RankineVortexFF(np.array((0.5, 0.5)), -1., 1.)

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "four_vortices":
            srf = 1.

            f = 1.
            fs = srf

            x_init = f * np.array([0., 0.])
            x_target = f * np.array([1., 0.])

            bl = f * np.array([-0.2, -1.])
            tr = f * np.array([1.2, 1.])

            omega = f * np.array(((0.5, 0.5),
                                  (0.5, 0.2),
                                  (0.5, -0.2),
                                  (0.5, -0.5)))
            strength = f * fs * np.array([1., -1., 1.5, -1.5])
            radius = f * np.array([1e-1, 1e-1, 1e-1, 1e-1])
            vortices = [RankineVortexFF(omega[i], strength[i], radius[i]) for i in range(len(omega))]
            const_wind = UniformFF(np.array([0., 0.]))

            ff = sum(vortices, const_wind)

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "moving_vortex":
            srf = 1.

            f = 1.
            fs = srf

            x_init = f * np.array([0., 0.])
            x_target = f * np.array([1., 0.])
            nt = 10
            xs = np.linspace(0.5, 0.5, nt)
            ys = np.linspace(-0.4, 0.4, nt)

            omega = f * np.stack((xs, ys), axis=1)
            gamma = f * fs * -1. * np.ones(nt)
            radius = f * 1e-1 * np.ones(nt)

            ff = RankineVortexFF(omega, gamma, radius, t_end=1. * f / srf)

            return cls(ff, x_init, x_target, srf, name=b_name)

        if b_name == "linear_time_varying":
            srf = 1.

            f = 1.
            x_init = f * np.array([0., 0.])
            x_target = f * np.array([1., 0.])

            gradient_init = np.array(((0., 3 * srf / f),
                                      (0., 0.)))

            gradient_diff_time = np.array(((0., -3 * srf / f),
                                           (0., 0.)))

            origin = np.array([0., 0.])
            value_origin = np.array([0., 0.])

            bl = f * np.array([-0.2, -1.])
            tr = f * np.array([1.2, 1.])

            ff = LinearFFT(gradient_diff_time, gradient_init, origin, value_origin)

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "moving_vortices":
            srf = 1.

            f = 1.
            fs = srf

            x_init = f * np.array([0., 0.])
            x_target = f * np.array([1., 0.])
            nt = 10
            g = f * fs * -1.
            r = f * 1e-3
            vortices = []
            xvors = np.array((0.2, 0.4, 0.6, 0.8))
            for k, xvor in enumerate(xvors):
                xs = np.linspace(xvor, xvor, nt)
                if k == 0:
                    ys = np.linspace(-0.3, 0.1, nt)
                elif k == 1:
                    ys = np.linspace(-0.1, 0.5, nt)
                elif k == 2:
                    ys = np.linspace(-0.2, 0.4, nt)[::-1]
                else:
                    ys = np.linspace(-0.4, 0.2, nt)[::-1]

                omega = f * np.stack((xs, ys), axis=1)
                gamma = g * np.ones(nt)
                radius = r * np.ones(nt)

                vortices.append(RankineVortexFF(omega, gamma, radius, t_end=1. * f / srf))

            obstacles = []
            # obstacles.append(RadialGaussFF(f * 0.5, f * 0.15, f * 0.1, 1 / 2 * 0.2, 10*v_a))
            ffs: list[FlowField] = [UniformFF(np.array((0., 0.)))] + vortices + obstacles
            # const_wind = UniformFF(np.array([0., 5.]))
            N = len(ffs) - len(obstacles)
            M = len(obstacles)

            coeffs = np.array(tuple(1 / N for _ in range(N)) + tuple(1. for _ in range(M)))
            ff = sum(list(map(lambda x: x[0] * x[1], zip(coeffs, ffs))), ZeroFF())

            return cls(ff, x_init, x_target, srf, name=b_name)

        if b_name == 'gyre_rhoads2010':
            srf = 1.
            x_init = np.array((0, 0.25))
            x_target = np.array((0.03, -0.25))
            bl = np.array((-1, -0.5))
            tr = np.array((1., 0.5))

            ff = GyreFF(0, -0.5, 2, 2, 2)
            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr)

        if b_name == "atlantic":
            ff = DiscreteFF.from_npz(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'demo', 'atlantic', 'flow.npz'))
            x_init = np.array((-0.7, 0.6))  # radians
            x_target = np.array((-0.15, 1.1))  # radians

            srf = 15.  # meters per second
            return cls(ff, x_init, x_target, srf, name=b_name)

        if b_name == "spherical_geodesic":
            x_init = np.array((0, np.pi / 6))
            x_target = np.array((np.pi / 4, np.pi / 6))
            # Only DiscreteFF can be GCS so discretize a null flow field
            ff = DiscreteFF.from_ff(ZeroFF(), (x_init - np.array((0.15, 0.15)), x_target + np.array((0.15, 0.15))),
                                    coords=Coords.GCS)
            srf = 1  # meters per second
            return cls(ff, x_init, x_target, srf, name=b_name)

        if b_name == "chertovskih":
            srf = 1.
            x_init = np.array((0.5, 0.))
            x_target = np.array((-0.7, -6))
            bl = np.array((-1.5, -6.2))
            tr = np.array((1.5, 0.2))
            ff = ChertovskihFF()
            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "dakar_natal_constr":
            v_a = 23.  # meters per seconds
            x_init = Utils.DEG_TO_RAD * np.array([-17.447938, 14.693425])
            x_target = Utils.DEG_TO_RAD * np.array([-35.2080905, -5.805398])
            ff = DiscreteFF.from_h5(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'demo', 'dakar_natal_constr', 'wind.h5'))
            return cls(ff, x_init, x_target, v_a, name=b_name)  # obstacles=obstacles)

        if b_name == "obstacle":
            srf = 1.
            x_init = np.array((0., 0.))
            x_target = np.array((1., 0.))
            ff = ZeroFF()
            obstacles = [CircleObs((0.5, 0.1), 0.2)]
            return cls(ff, x_init, x_target, srf, obstacles=obstacles, name=b_name)

        if b_name == "trap":
            srf = 1.
            x_init = np.array((0., 0.))
            x_target = np.array((1., 0.))
            bl = np.array((-0.2, -0.6))
            tr = np.array((1.2, 0.6))

            nt = 40
            wind_value = 2 * np.ones(nt)
            center = np.zeros((nt, 2))
            for k in range(30):
                center[5 + k] = np.array((0.05 * k, 0.))
            radius = 0.2 * np.ones(nt)

            ff = TrapFF(wind_value, center, radius, t_end=4)
            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "stream":
            srf = 1
            x_init = np.array((1, 1))
            x_target = np.array((4, 4))
            bl = np.array((0.75, 0.75))
            tr = np.array((4.25, 4.25))

            ff = BandFF(np.array((0., 2.5)), np.array((1., 0.)), np.array((-1., 0.)), 1)
            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "montreal_reykjavik":
            ff = DiscreteFF.from_npz(os.path.join(os.path.abspath('..'), 'data', 'demo', 'montreal_reykjavik',
                                                  'ff.npz'))
            x_init = np.diag((5 / 6, 1 / 2)) @ (ff.bounds[1:, 1] - ff.bounds[1:, 0])
            x_target = np.diag((1 / 6, 1 / 2)) @ (ff.bounds[1:, 1] - ff.bounds[1:, 0])
            return cls(ff, x_init, x_target, 10, name=b_name)

        if b_name == "reykjavik_dublin":
            ff = DiscreteFF.from_npz(os.path.join(os.path.abspath('..'), 'data', 'cds_omerc', 'reykjavik_dublin',
                                                  'ff.npz'))
            x_init = np.diag((5 / 6, 1 / 2)) @ (ff.bounds[1:, 1] - ff.bounds[1:, 0])
            x_target = np.diag((1 / 6, 1 / 2)) @ (ff.bounds[1:, 1] - ff.bounds[1:, 0])
            return cls(ff, x_init, x_target, 10, name=b_name)

        if b_name == "double_gyre_time_dependent":
            ff = GyreMSEASFF()
            x_init = np.array((0.25, 0.25))
            x_target = np.array((1.75, 0.75))
            bl = np.array((0, 0))
            tr = np.array((2, 1))
            return cls(ff, x_init, x_target, 0.5, name=b_name, bl=bl, tr=tr)

        else:
            raise ValueError('No corresponding problem for name "%s"' % b_name)


def timeopt_control_cartesian(costate: ndarray, srf_max: float):
    return -srf_max * costate / np.linalg.norm(costate)


def timeopt_control_gcs(state: ndarray, costate: ndarray, srf_max: float):
    costate_mod = costate.dot(np.array(((1 / np.cos(state[..., 1]), 0.), (0., 1.))))
    return -srf_max * costate_mod / np.linalg.norm(costate_mod)
