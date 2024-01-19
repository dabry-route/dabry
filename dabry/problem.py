import json
import math
import os
import warnings
from datetime import datetime
from typing import Optional, List

import numpy as np
from numpy import ndarray, pi

from dabry.aero import MermozAero, Aero
from dabry.ddf_manager import DDFmanager
from dabry.feedback import GSTargetFB
from dabry.flowfield import RankineVortexFF, UniformFF, DiscreteFF, StateLinearFF, RadialGaussFF, GyreFF, \
    PointSymFF, LinearFFT, BandFF, TrapFF, ChertovskihFF, \
    FlowField, VortexFF, ZeroFF, WrapperFF
from dabry.misc import Utils, csv_to_dict
from dabry.model import Model
from dabry.obstacle import CircleObs, FrameObs, GreatCircleObs, ParallelObs, MeridianObs, Obstacle, WrapperObs
from dabry.penalty import Penalty, CirclePenalty, NullPenalty, WrapperPen

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
                 bl: Optional[ndarray] = None, tr: Optional[ndarray] = None,
                 name: Optional[str] = None, aero: Optional[Aero] = None, penalty: Optional[Penalty] = None,
                 autoframe=True):
        self.model = Model.zermelo(ff)
        self.x_init: ndarray = x_init.copy()
        self.x_target: ndarray = x_target.copy()
        self.srf_max: float = srf_max
        self.aero: Aero = MermozAero() if aero is None else aero

        # Domain bounding box corners
        self.bl: ndarray = bl.copy() if bl is not None else np.array(())
        self.tr: ndarray = tr.copy() if tr is not None else np.array(())

        self.length_reference = np.min(bl - tr) if bl is not None and tr is not None else \
            np.linalg.norm(x_target - x_init)

        self.name = 'Unnamed problem' if name is None else name

        self.obstacles: list[Obstacle] = obstacles if obstacles is not None else []
        self.penalty: Penalty = NullPenalty() if penalty is None else penalty

        self.io = DDFmanager()

        # Bound computation domain on wind grid limits
        if self.bl.size == 0 or self.tr.size == 0:
            ff = self.model.ff
            if type(ff) == DiscreteFF:
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

        frame_offset = 0.
        if autoframe:
            bl_frame = self.bl + (self.tr - self.bl) * frame_offset / 2.
            tr_frame = self.tr - (self.tr - self.bl) * frame_offset / 2.
            self.obs_frame = FrameObs(bl_frame, tr_frame)
            self.obstacles.append(self.obs_frame)
        # self.obstacles.extend(NavigationProblem.spherical_frame(bl, tr, offset_rel=0))

    def update_ff(self, ff: FlowField):
        self.model = Model.zermelo(ff)

    def save_ff(self):
        self.io.dump_ff(self.model.ff, bl=self.bl, tr=self.tr)

    def save_info(self):
        pb_info = {
            'x_init': tuple(self.x_init),
            'x_target': tuple(self.x_target),
            'airspeed': self.srf_max,
            'geodesic_time_no_ff': self.geod_l / self.srf_max,
            'geodesic_distance': self.geod_l,
            'aero_mode': self.aero.mode
        }
        with open(os.path.join(self.io.case_dir, 'pb_info.json'), 'w') as f:
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
        return -self.srf_max * costate / np.linalg.norm(costate)

    def timeopt_control_gcs(self, state: ndarray, costate: ndarray):
        costate_mod = costate.dot(np.array(((1 / np.cos(state[..., 1]), 0.), (0., 1.))))
        return -self.srf_max * costate_mod / np.linalg.norm(costate_mod)

    def augsys_dyn_timeopt(self, t: float, state: ndarray, costate: ndarray, control: ndarray):
        return np.hstack((self.model.dyn.value(t, state, control),
                          -self.model.dyn.d_value__d_state(t, state, control).transpose() @ costate
                          - self.penalty.d_value(t, state)))

    def augsys_dyn_timeopt_cartesian(self, t: float, state: ndarray, costate: ndarray):
        return self.augsys_dyn_timeopt(t, state, costate, self.timeopt_control_cartesian(costate))

    def augsys_dyn_timeopt_gcs(self, t: float, state: ndarray, costate: ndarray):
        return self.augsys_dyn_timeopt(t, state, costate, self.timeopt_control_gcs(state, costate))

    def control_angle(self, adjoint, state=None):
        if self.coords == Utils.COORD_CARTESIAN:
            # angle to x-axis in trigonometric angle
            return np.arctan2(*(-adjoint)[::-1])
        else:
            # gcs, angle from north in cw order
            mat = np.diag((1 / np.cos(state[1]), 1.))
            pl = mat @ adjoint
            return np.pi / 2. - np.arctan2(*-pl[::-1])

    def in_obs(self, x):
        obs_list = []
        res = False
        for i, obs in enumerate(self.obstacles):
            val = obs.value(x)
            if val < 0.:
                res = True
                obs_list.append(i)
        if not res:
            return [-1]
        else:
            return obs_list

    def orthodromic(self):
        fb = GSTargetFB(self.model.ff, self.srf_max, self.x_target)
        f = lambda x, t: self.model.dyn(t, x, fb.value(t, x))
        t_max = 2 * self.time_scale
        # res = scitg.odeint(f, self.x_init, np.linspace(0, t_max, 100))
        # TODO continue implementation
        return -1

    def non_dimensionalize(self):
        if self.coords == Utils.COORD_CARTESIAN:
            scale_length = self.length_reference
            x_init = (self.x_init - self.bl) / scale_length
            x_target = (self.x_target - self.bl) / scale_length
            bl_pb_adim = np.zeros(2)
            bl_wrappers = self.bl.copy()
            tr_pb_adim = (self.tr - self.bl) / scale_length
            srf_max = self.srf_max
        else:
            # Means self.coords == Utils.COORD_GCS:
            # Do not rescale lon/lat because computation will be false
            scale_length = 1.
            x_init = self.x_init
            x_target = self.x_target
            bl_pb_adim = self.bl
            bl_wrappers = np.zeros(2)
            tr_pb_adim = self.tr
            # In GCS convention, srf is specified in meters per seconds and
            # has to be cast to radians per seconds before regular scaling
            srf_max = self.srf_max / (Utils.EARTH_RADIUS if self.coords == Utils.COORD_GCS else 1.)
            # srf is now either m/s or rad/s and length_reference is either m or rad
            # so build time scale over it

        scale_time = scale_length / srf_max
        # In the new system, srf_max is unit
        srf_max = 1.

        wrapper_ff = WrapperFF(self.model.ff, scale_length, bl_wrappers, scale_time, self.model.ff.t_start)

        obstacles = self.obstacles.copy()
        obstacles.remove(self.obs_frame)
        obstacles = [WrapperObs(obs, self.length_reference, self.bl) for obs in obstacles]
        penalty = WrapperPen(self.penalty, self.length_reference, self.bl, scale_time, self.model.ff.t_start)
        return NavigationProblem(wrapper_ff, x_init, x_target, srf_max,
                                 bl=bl_pb_adim, tr=tr_pb_adim, obstacles=obstacles, penalty=penalty)

    # TODO: reimplement this
    def htarget(self):
        # fb = HTargetFB(self.x_target, self.coords)
        # self.load_feedback(fb)
        # sc = DistanceSC(lambda x: self.distance(x, self.x_target), self.geod_l * 0.01)
        # traj = self.integrate_trajectory(self.x_init, sc, int_step=self.time_scale / 2000, max_iter=6000,
        #                                  t_init=self.model.ff.t_start)
        # if sc.value(0, traj.points[traj.last_index]):
        #     return traj.timestamps[traj.last_index] - self.model.ff.t_start
        # else:
        #     return -1
        return -1

    @classmethod
    def from_database(cls, x_init: ndarray, x_target: ndarray, srf: float,
                      t_start: [float, datetime], t_end: [float, datetime], obstacles: Optional[List[Obstacle]] = None,
                      resolution='0.5', pressure_level='1000', data_path: Optional[str] = None):
        """
        :param x_init: Initial position (lon, lat)
        :param x_target: Target position (lon, lat)
        :param srf: Speed relative to flow
        :param t_start: Time window lower bound
        :param t_end: Time window upper bound
        :param obstacles: List of obstacle objects
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

        if obstacles is None:
            obstacles = []

        obstacles = obstacles + NavigationProblem.spherical_frame(bl, tr)

        return cls(ff, x_init, x_target, srf, obstacles=obstacles, bl=bl, tr=tr)

    @staticmethod
    def spherical_frame(bl, tr, offset_rel=0.02):
        offset = (tr - bl) * offset_rel
        bl_obs = bl + offset
        tr_obs = tr - offset

        obstacles = []
        obstacles.append(MeridianObs(bl_obs[0], True))
        obstacles.append(MeridianObs(tr_obs[0], False))
        obstacles.append(ParallelObs(bl_obs[1], True))
        obstacles.append(ParallelObs(tr_obs[1], False))
        eps_lon = Utils.angular_diff(bl_obs[0], tr_obs[0]) / 30
        eps_lat = Utils.angular_diff(bl_obs[1], tr_obs[1]) / 30
        obstacles.append(GreatCircleObs(np.array((bl_obs[0], bl_obs[1])) + np.array((eps_lon, 0)),
                                        np.array((bl_obs[0], bl_obs[1])) + np.array((0, eps_lat)), autobox=True))
        obstacles.append(GreatCircleObs(np.array((tr_obs[0], bl_obs[1])) + np.array((0, eps_lat)),
                                        np.array((tr_obs[0], bl_obs[1])) + np.array((-eps_lon, 0)), autobox=True))
        obstacles.append(GreatCircleObs(np.array((tr_obs[0], tr_obs[1])) + np.array((-eps_lon, 0)),
                                        np.array((tr_obs[0], tr_obs[1])) + np.array((0, -eps_lat)), autobox=True))
        obstacles.append(GreatCircleObs(np.array((bl_obs[0], tr_obs[1])) + np.array((0, -eps_lat)),
                                        np.array((bl_obs[0], tr_obs[1])) + np.array((eps_lon, 0)), autobox=True))
        obstacles.append(ParallelObs(-80 * Utils.DEG_TO_RAD, True))
        obstacles.append(ParallelObs(80 * Utils.DEG_TO_RAD, False))
        return obstacles

    @classmethod
    def base_name(cls, name: str):
        for b_name, attrs in cls.ALL.items():
            if b_name == name:
                return name
            if attrs['s_name'] == name:
                return b_name
        raise ValueError('Cannot find problem "%s". Check problems.csv' % name)

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

            bl = np.array([-1, -1])
            tr = np.array([2, 2])

            ff = sum([VortexFF(np.array((0.5, 0.8)), -1.),
                      VortexFF(np.array((0.8, 0.2)), -0.8),
                      VortexFF(np.array((0.6, -0.5)), 0.8)], ZeroFF())

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

        if b_name == "gyre":
            srf = 0.05

            sf = 1.

            x_init = sf * np.array((0.6, 0.6))
            x_target = sf * np.array((2.4, 2.4))
            bl = sf * np.array((0.5, 0.5))
            tr = sf * np.array((2.5, 2.5))

            ff = GyreFF(0.5, 0.5, 2., 2., pi * 0.02)

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr, name=b_name)

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
            srf = 1.

            f = 1.
            fs = srf

            x_init = f * np.array([0., 0.])
            x_target = f * np.array([1., 0.])

            bl = f * np.array([-0.2, -1.])
            tr = f * np.array([1.2, 1.])
            omega = f * np.array(((0.2, -0.2), (0.8, 0.2)))

            vortex = [
                RankineVortexFF(omega[0], f * fs * -7., f * 1.),
                RankineVortexFF(omega[1], f * fs * -7., f * 1.)
            ]

            ff = vortex[0] + vortex[1]

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

            return cls(ff, x_init, x_target, srf)

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

            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr)

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
            ffs: list[FlowField] = [UniformFF(np.array((-5., 0.)))] + vortices + obstacles
            # const_wind = UniformFF(np.array([0., 5.]))
            N = len(ffs) - len(obstacles)
            M = len(obstacles)

            coeffs = np.array(tuple(1 / N for _ in range(N)) + tuple(1. for _ in range(M)))
            ff = sum(list(map(lambda x: x[0] * x[1], zip(coeffs, ffs))), ZeroFF())

            return cls(ff, x_init, x_target, srf)

        if b_name == 'gyre_rhoads2010':
            v_a = 1.
            sf = 1.
            x_init = sf * np.array((0, 0.25))
            x_target = sf * np.array((0.03, -0.25))
            bl = sf * np.array((-1, -0.5))
            tr = sf * np.array((1., 0.5))

            ff = GyreFF(sf * 0, sf * -0.5, sf * 2, sf * 2, 30.)
            return cls(ff, x_init, x_target, v_a, bl=bl, tr=tr)

        if b_name == "atlantic":
            # TODO: get a well traced gcs wind test case
            ff = DiscreteFF.from_npz(os.path.join(os.path.dirname(__file__), '..', 'data_demo',
                                                  'atlantic', 'flow.npz'))
            x_init = np.array((-0.7, 0.6))  # radians
            x_target = np.array((-0.15, 1.1))  # radians

            srf = 15.  # meters per second
            return cls(ff, x_init, x_target, srf)

        if b_name == "spherical_geodesic":
            x_init = np.array((0, np.pi / 6))
            x_target = np.array((np.pi / 4, np.pi / 6))
            # Only DiscreteFF can be GCS so discretize a null flow field
            ff = DiscreteFF.from_ff(ZeroFF(), (x_init - np.array((0.15, 0.15)), x_target + np.array((0.15, 0.15))),
                                    coords='gcs')
            srf = 1  # meters per second
            return cls(ff, x_init, x_target, srf)

        if b_name == "chertovskih":
            srf = 1.
            x_init = np.array((0.5, 0.))
            x_target = np.array((-0.7, -6))
            bl = np.array((-1.5, -6.2))
            tr = np.array((1.5, 0.2))
            ff = ChertovskihFF()
            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr)

        if b_name == "dakar_natal_constr":
            v_a = 23.
            x_init = Utils.DEG_TO_RAD * np.array([-17.447938, 14.693425])
            x_target = Utils.DEG_TO_RAD * np.array([-35.2080905, -5.805398])
            ff = DiscreteFF.from_h5(os.path.join(os.path.dirname(__file__), '..',
                                                 'data_demo/ncdc/44W_16S_9W_25N_20210929_00/wind.h5'))
            # obstacles = [LSEMaxiObs([
            #     GreatCircleObs(Utils.DEG_TO_RAD * np.array((-17, 10)),
            #                    Utils.DEG_TO_RAD * np.array((-30, 15))),
            #     GreatCircleObs(Utils.DEG_TO_RAD * np.array((-20, 5)),
            #                    Utils.DEG_TO_RAD * np.array((-17, 10))),
            #     GreatCircleObs(Utils.DEG_TO_RAD * np.array((-30, 7)),
            #                    Utils.DEG_TO_RAD * np.array((-20, 5)))
            # ])]
            return cls(ff, x_init, x_target, v_a, )  # obstacles=obstacles)

        if b_name == "obstacle":
            srf = 1.
            x_init = np.array((0., 0.))
            x_target = np.array((1., 0.))
            ff = ZeroFF()
            obstacles = [CircleObs((0.5, 0.1), 0.2)]
            return cls(ff, x_init, x_target, srf, obstacles=obstacles)

        if b_name == "trap":
            # TODO: adjust coefficients
            srf = 1.
            x_init = np.array((0., 0.))
            x_target = np.array((1., 0.))
            bl = np.array((-0.2, -0.6))
            tr = np.array((1.2, 0.6))

            nt = 40
            wind_value = 4 * np.ones(nt)
            center = np.zeros((nt, 2))
            for k in range(30):
                center[10 + k] = np.array((0.05 * k, 0.))
            radius = 0.2 * np.ones(nt)

            ff = TrapFF(wind_value, center, radius, t_end=4)
            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr)

        if b_name == "stream":
            # TODO: fix case
            srf = 1.
            sf = 1.
            x_init = sf * np.array((1, 1))
            x_target = sf * np.array((4, 4))
            bl = sf * np.array((0.75, 0.75))
            tr = sf * np.array((4.25, 4.25))

            band_wind = BandFF(np.array((0., 2.5)), np.array((1., 0.)), np.array((-1., 0.)), 1)
            ff = DiscreteFF.from_ff(band_wind, np.array((bl, tr)), force_no_diff=True)
            ff.compute_derivatives()
            return cls(ff, x_init, x_target, srf, bl=bl, tr=tr)


class NPSanjuanDublinOrthoTV(NavigationProblem):
    def __init__(self):
        v_a = 23.
        x_init = np.array((1.6e6, 1.6e6))
        x_target = np.array((-1.8e6, 0.5e6))
        bl = np.array((-2.3e6, -1.5e6))
        tr = np.array((2e6, 2e6))
        ff = DiscreteFF.from_h5(os.path.join(os.environ.get('DABRYPATH'),
                                             'data_demo/ncdc/san-juan-dublin-flattened-ortho-tv.mz/wind.h5'))

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr, autoframe=True)
