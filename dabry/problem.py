import json
import math
import os
from datetime import datetime
from typing import Optional, List

import numpy as np
import scipy.integrate as scitg
from numpy import ndarray, pi

from dabry.aero import MermozAero, Aero
from dabry.ddf_manager import DDFmanager
from dabry.feedback import GSTargetFB
from dabry.misc import Utils
from dabry.model import Model
from dabry.obstacle import CircleObs, FrameObs, GreatCircleObs, ParallelObs, MeridianObs, Obstacle, MeanObs, LSEMaxiObs
from dabry.penalty import Penalty
from dabry.flowfield import RankineVortexFF, UniformFF, DiscreteFF, LinearFF, RadialGaussFF, DoubleGyreFF, \
    PointSymFF, LCFF, LinearFFT, BandFF, TrapFF, ChertovskihFF, \
    FlowField

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

    def __init__(self, ff: FlowField, x_init: ndarray, x_target: ndarray, srf_max: float,
                 obstacles: Optional[List[Obstacle]] = None,
                 bl: Optional[ndarray] = None, tr: Optional[ndarray] = None,
                 name: Optional[str] = None, time_scale: Optional[float] = None,
                 aero: Optional[Aero] = None, penalty: Optional[Penalty] = None,
                 autoframe=True):
        self.model = Model.zermelo(ff)
        self.x_init: ndarray = x_init.copy()
        self.x_target: ndarray = x_target.copy()
        self.srf_max = srf_max
        # self.aero = LLAero(mode='dabry')
        self.aero = MermozAero() if aero is None else aero

        # Domain bounding box corners
        self.bl: Optional[ndarray] = bl.copy() if bl is not None else None
        self.tr: Optional[ndarray] = tr.copy() if tr is not None else None

        self.geod_l = Utils.distance(self.x_init, self.x_target, coords=self.coords)

        # It is usually sufficient to scale time on geodesic / srf
        # but for some problems it isn't
        self.time_scale = self.geod_l / self.srf_max if time_scale is None else time_scale

        self.descr = 'Unnamed problem' if name is None else name

        # Bound computation domain on wind grid limits
        if bl is None or tr is None:
            ff = self.model.ff
            if type(ff) == DiscreteFF:
                self.bl = np.array((ff.bounds[0, 0], ff.bounds[1, 0]))
                self.tr = np.array((ff.bounds[0, 1], ff.bounds[1, 1]))
            else:
                w = 1.15 * self.geod_l
                self.bl = (self.x_init + self.x_target) / 2. - np.array((w / 2., w / 2.))
                self.tr = (self.x_init + self.x_target) / 2. + np.array((w / 2., w / 2.))
        if self.coords == Utils.COORD_GCS:
            self._domain_obs = NavigationProblem.spherical_frame(bl, tr, offset_rel=0)
            self.domain = lambda x: np.all([obs.value(x) > 0. for obs in self._domain_obs])
        else:
            self.domain = lambda x: self.bl[0] < x[0] < self.tr[0] and self.bl[1] < x[1] < self.tr[1]

        if self.bl is not None and self.tr is not None:
            self.l_ref = self.distance(self.bl, self.tr)
        else:
            self.l_ref = self.geod_l

        self.obstacles = obstacles if obstacles is not None else []
        if len(self.obstacles) > 0:
            for obs in self.obstacles:
                obs.update_lref(self.l_ref)

        if penalty is None:
            self.penalty = Penalty(lambda _, __: 0., lambda _, __: np.array((0, 0)))
        else:
            self.penalty = penalty

        self.frame_offset = 0.05
        if autoframe:
            bl_frame = self.bl + (self.tr - self.bl) * self.frame_offset / 2.
            tr_frame = self.tr - (self.tr - self.bl) * self.frame_offset / 2.
            self.obstacles.append(FrameObs(bl_frame, tr_frame))

        self.io = DDFmanager()

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
        return str(self.descr)

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

        for attr in ['model', 'geod_l', 'descr', 'domain', 'l_ref', 'frame_offset', 'io', '_domain_obs']:
            del kwargs[attr]

        return NavigationProblem(**kwargs)

    def distance(self, x1, x2):
        return Utils.distance(x1, x2, self.coords)

    def middle(self, x1, x2):
        return Utils.middle(x1, x2, self.coords)

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


class NP3vor(NavigationProblem):
    def __init__(self):
        v_a = 14.11

        f = 1.
        fs = 15.46

        x_init = f * np.array([0., 0.])
        x_target = f * np.array([1., 0.])

        bl = f * np.array([-1, -1])
        tr = f * np.array([2, 2])

        omega1 = f * np.array((0.5, 0.8))
        omega2 = f * np.array((0.8, 0.2))
        omega3 = f * np.array((0.6, -0.5))

        # import csv
        # with open('/dabry/.seeds/problem0/center1.csv', 'r') as file:
        #     reader = csv.reader(file)
        #     for k, row in enumerate(reader):
        #         if k == seed:
        #             omega1 = f * np.array(list(map(float, row)))
        #             break
        # with open('/dabry/.seeds/problem0/center2.csv', 'r') as file:
        #     reader = csv.reader(file)
        #     for k, row in enumerate(reader):
        #         if k == seed:
        #             omega2 = f * np.array(list(map(float, row)))
        #             break
        # with open('/dabry/.seeds/problem0/center3.csv', 'r') as file:
        #     reader = csv.reader(file)
        #     for k, row in enumerate(reader):
        #         if k == seed:
        #             omega3 = f * np.array(list(map(float, row)))
        #             break

        vortex1 = RankineVortexFF(omega1, f * fs * -1., f * 1e-1)
        vortex2 = RankineVortexFF(omega2, f * fs * -0.8, f * 1e-1)
        vortex3 = RankineVortexFF(omega3, f * fs * 0.8, f * 1e-1)
        const_wind = UniformFF(np.array([0., 0.]))

        ff = LCFF(np.array((3., 1., 1., 1.)),
                    (const_wind, vortex1, vortex2, vortex3))

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NPLinear(NavigationProblem):
    def __init__(self):
        v_a = 23.

        f = 1e6
        x_init = f * np.array([0., 0.])
        x_target = f * np.array([1., 0.])

        gradient = np.array([[0., v_a / f], [0., 0.]])
        origin = np.array([0., 0.])
        value_origin = np.array([0., 0.])

        bl = f * np.array([-0.2, -1.])
        tr = f * np.array([1.2, 1.])

        ff = LinearFF(gradient, origin, value_origin)

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NPDoubleGyreLi(NavigationProblem):
    def __init__(self):
        v_a = 0.6

        sf = 1.

        x_init = sf * np.array((125., 125.))
        x_target = sf * np.array((375., 375.))
        bl = sf * np.array((-10, -10))
        tr = sf * np.array((510, 510))

        ff = DoubleGyreFF(0., 0., 500., 500., 1.)

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NPDoubleGyreKularatne(NavigationProblem):
    def __init__(self):
        v_a = 0.05

        sf = 1.

        x_init = sf * np.array((0.6, 0.6))
        x_target = sf * np.array((2.4, 2.4))
        bl = sf * np.array((0.5, 0.5))
        tr = sf * np.array((2.5, 2.5))

        ff = DoubleGyreFF(0.5, 0.5, 2., 2., pi * 0.02)

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NPPointSymTechy(NavigationProblem):
    def __init__(self):
        v_a = 18.
        sf = 1e6
        x_init = sf * np.array((0., 0.))
        x_target = sf * np.array((1., 0.))
        bl = sf * np.array((-0.1, -1.))
        tr = sf * np.array((1.1, 1.))

        # To get w as wind value at start point, choose gamma = w / 0.583
        gamma = v_a / sf * 1.
        omega = 0.
        ff = PointSymFF(sf * 0.5, sf * 0.3, gamma, omega)

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NP3obs(NavigationProblem):
    def __init__(self):
        v_a = 23.
        sf = 3e6

        x_init = sf * np.array((0., 0.))
        x_target = sf * np.array((1., 0.))
        bl = sf * np.array((-0.15, -1.15))
        tr = sf * np.array((1.15, 1.15))

        const_wind = UniformFF(np.array([1., 1.]))

        c1 = sf * np.array((0.5, 0.))
        c2 = sf * np.array((0.5, 0.1))
        c3 = sf * np.array((0.5, -0.1))

        wind_obstacle1 = RadialGaussFF(c1[0], c1[1], sf * 0.1, 1 / 2 * 0.2, v_a * 5.)
        wind_obstacle2 = RadialGaussFF(c2[0], c2[1], sf * 0.1, 1 / 2 * 0.2, v_a * 5.)
        wind_obstacle3 = RadialGaussFF(c3[0], c3[1], sf * 0.1, 1 / 2 * 0.2, v_a * 5.)

        ff = wind_obstacle1 + wind_obstacle2 + wind_obstacle3 + const_wind

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NPSanjuanDublinOrtho(NavigationProblem):
    def __init__(self):
        v_a = 23.
        ff = DiscreteFF.from_h5(os.path.join(os.environ.get('DABRYPATH'),
                                               'data_demo/ncdc/san-juan-dublin-flattened-ortho.mz/wind.h5'))

        # point = np.array([-66.116666, 18.465299])
        # x_init = np.array(proj(*point)) + 500e3 * np.ones(2)
        x_init = np.array((-5531296, 2020645))

        # point = np.array([-6.2602732, 53.3497645])
        # x_target = np.array(proj(*point)) - 500e3 * np.ones(2)
        x_target = np.array((-415146, 511759))

        super().__init__(ff, x_init, x_target, v_a)


class NPBigRankine(NavigationProblem):
    def __init__(self):
        v_a = 23.

        f = 1e6
        fs = v_a

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

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NP4vor(NavigationProblem):
    def __init__(self):
        v_a = 23.

        f = 1e6
        fs = v_a

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

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NPMovor(NavigationProblem):
    def __init__(self):
        v_a = 23.

        f = 1e6
        fs = v_a

        x_init = f * np.array([0., 0.])
        x_target = f * np.array([1., 0.])
        nt = 10
        xs = np.linspace(0.5, 0.5, nt)
        ys = np.linspace(-0.4, 0.4, nt)

        omega = f * np.stack((xs, ys), axis=1)
        gamma = f * fs * -1. * np.ones(nt)
        radius = f * 1e-1 * np.ones(nt)

        ff = RankineVortexFF(omega, gamma, radius, t_end=1. * f / v_a)

        super().__init__(ff, x_init, x_target, v_a)


class NPTVLinear(NavigationProblem):
    def __init__(self):
        v_a = 23.

        f = 1e6
        x_init = f * np.array([0., 0.])
        x_target = f * np.array([1., 0.])

        nt = 20

        gradient = np.array([[np.zeros(nt), np.linspace(3 * v_a / f, 0 * v_a / f, nt)],
                             [np.zeros(nt), np.zeros(nt)]]).transpose((2, 0, 1))

        origin = np.array([0., 0.])
        value_origin = np.array([0., 0.])

        bl = f * np.array([-0.2, -1.])
        tr = f * np.array([1.2, 1.])

        ff = LinearFFT(gradient, origin, value_origin, t_end=0.5 * f / v_a)

        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NPMovors(NavigationProblem):
    def __init__(self):
        v_a = 23.

        f = 1e6
        fs = v_a

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

            vortices.append(RankineVortexFF(omega, gamma, radius, t_end=1. * f / v_a))

        obstacles = []
        # obstacles.append(RadialGaussFF(f * 0.5, f * 0.15, f * 0.1, 1 / 2 * 0.2, 10*v_a))
        winds = [UniformFF(np.array((-5., 0.)))] + vortices + obstacles
        # const_wind = UniformFF(np.array([0., 5.]))
        N = len(winds) - len(obstacles)
        M = len(obstacles)

        ff = LCFF(np.array(tuple(1 / N for _ in range(N)) + tuple(1. for _ in range(M))), tuple(winds))

        super().__init__(ff, x_init, x_target, v_a)


class NPGyreRhoads(NavigationProblem):
    def __init__(self):
        v_a = 23.
        sf = 3e6
        x_init = sf * np.array((0, 0.25))
        x_target = sf * np.array((0.03, -0.25))
        bl = sf * np.array((-1, -0.5))
        tr = sf * np.array((1., 0.5))

        ff = DoubleGyreFF(sf * 0, sf * -0.5, sf * 2, sf * 2, 30.)
        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr, time_scale=3. * self.geod_l / v_a)


class NPBandFF(NavigationProblem):
    def __init__(self):
        v_a = 20.7
        sf = 1.
        x_init = sf * np.array((20., 20.))
        x_target = sf * np.array((80., 80.))
        bl = sf * np.array((15, 15))
        tr = sf * np.array((85, 85))

        band_wind = BandFF(np.array((0., 50.)), np.array((1., 0.)), np.array((-20., 0.)), 20)
        ff = DiscreteFF.from_ff(band_wind, np.array((bl, tr)))
        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NPTrapFF(NavigationProblem):
    def __init__(self):
        v_a = 23.
        sf = 3e6
        x_init = sf * np.array((0., 0.))
        x_target = sf * np.array((1., 0.))
        bl = sf * np.array((-0.2, -0.6))
        tr = sf * np.array((1.2, 0.6))

        nt = 40
        wind_value = 80 * np.ones(nt)
        center = np.zeros((nt, 2))
        for k in range(30):
            center[10 + k] = sf * np.array((0.05 * k, 0.))
        radius = sf * 0.2 * np.ones(nt)

        ff = TrapFF(wind_value, center, radius, t_end=400000)
        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


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


class NPObs(NavigationProblem):
    def __init__(self):
        v_a = 23.
        sf = 3e6
        x_init = sf * np.array((0.1, 0.))
        x_target = sf * np.array((0.9, 0.))
        ff = UniformFF(np.array((5., 5.)))

        obstacles = []
        # obstacles.append(CircleObs(sf * np.array((0.4, 0.1)), sf * 0.1))
        # obstacles.append(CircleObs(sf * np.array((0.5, 0.)), sf * 0.1))
        obstacles.append(MeanObs([CircleObs(sf * np.array((0.4, 0.1)), sf * 0.1),
                                  CircleObs(sf * np.array((0.5, 0.)), sf * 0.1)]))
        obstacles.append(FrameObs(sf * np.array((0.05, -0.45)), sf * np.array((0.95, 0.45))))

        super().__init__(ff, x_init, x_target, v_a, obstacles=obstacles, autoframe=False)


class NPChertovskih(NavigationProblem):
    def __init__(self):
        v_a = 1.
        x_init = np.array((0.5, 0.))
        x_target = np.array((-0.7, -6))
        bl = np.array((-1.5, -6.2))
        tr = np.array((1.5, 0.2))
        ff = ChertovskihFF()
        super().__init__(ff, x_init, x_target, v_a, bl=bl, tr=tr)


class NPDakarNatalConstr(NavigationProblem):
    def __init__(self):
        v_a = 23.
        x_init = Utils.DEG_TO_RAD * np.array([-17.447938, 14.693425])
        x_target = Utils.DEG_TO_RAD * np.array([-35.2080905, -5.805398])
        ff = DiscreteFF.from_h5(os.path.join(os.environ.get('DABRYPATH'),
                                               'data_demo/ncdc/44W_16S_9W_25N_20210929_00/wind.h5'))
        obstacles = [LSEMaxiObs([
            GreatCircleObs(Utils.DEG_TO_RAD * np.array((-17, 10)),
                           Utils.DEG_TO_RAD * np.array((-30, 15))),
            GreatCircleObs(Utils.DEG_TO_RAD * np.array((-20, 5)),
                           Utils.DEG_TO_RAD * np.array((-17, 10))),
            GreatCircleObs(Utils.DEG_TO_RAD * np.array((-30, 7)),
                           Utils.DEG_TO_RAD * np.array((-20, 5)))
        ])]
        super().__init__(ff, x_init, x_target, v_a, obstacles=obstacles)


all_problems = {'3vor': (NP3vor, 'Three vortices'),
                'linear': (NPLinear, 'Linear wind'),
                'double-gyre-li2020': (NPDoubleGyreLi, 'Double gyre Li 2020'),
                'double-gyre-ku2016': (NPDoubleGyreKularatne, 'Double gyre Kularatne 2016'),
                'pointsym-techy2011': (NPPointSymTechy, 'Point symmetric Techy 2011'),
                '3obs': (NP3obs, 'Three obstacles'),
                'sanjuan-dublin-ortho': (NPSanjuanDublinOrtho, 'San-Juan Dublin Orthographic proj'),
                'big_rankine': (NPBigRankine, 'Big Rankine vortex'),
                '4vor': (NP4vor, 'Four vortices'),
                'movor': (NPMovor, 'One moving vortex'),
                'tvlinear': (NPTVLinear, 'Linear flow field varying in time'),
                'movors': (NPMovors, 'Multiple moving small vortices'),
                'gyre-rhoads2010': (NPGyreRhoads, 'Gyre flow field Rhoads 2010'),
                'band': (NPBandFF, 'Band of flow field'),
                'trap': (NPTrapFF, 'Trap of flow field'),
                'sanjuan-dublin-ortho-tv': (NPSanjuanDublinOrthoTV, 'San-Juan Dublin Ortho proj Unsteady'),
                'obs': (NPObs, 'Obstacle'),
                'chertovskih2020': (NPChertovskih, 'Chertovskih 2020'),
                'dakar-natal-constr': (NPDakarNatalConstr, 'Dakar Natal Constrained')}
