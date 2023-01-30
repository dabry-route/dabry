import h5py
from pyproj import Proj
import numpy as np
from numpy import ndarray, pi

from dabry.aero import LLAero, MermozAero
from dabry.feedback import Feedback, AirspeedLaw, MultiFeedback, GSTargetFB
from dabry.integration import IntEulerExpl
from dabry.wind import RankineVortexWind, UniformWind, DiscreteWind, LinearWind, RadialGaussWind, DoubleGyreWind, \
    PointSymWind, BandGaussWind, RadialGaussWindT, LCWind, LinearWindT, BandWind, LVWind, TrapWind, ChertovskihWind
from dabry.model import Model, ZermeloGeneralModel
from dabry.stoppingcond import StoppingCond, TimedSC, DistanceSC
from dabry.misc import Utils
from dabry.obstacle import CircleObs, FrameObs, GreatCircleObs, ParallelObs, MeridianObs, LSEMaxiObs, MeanObs

"""
problem.py
Handles navigation problem data
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

    def __init__(self,
                 model: Model,
                 x_init,
                 x_target,
                 coords,
                 domain=None,
                 obstacles=None,
                 bl=None,
                 tr=None,
                 autodomain=True,
                 autoframe=False,
                 descr=None,
                 time_scale=None,
                 **kwargs):
        self.model = model
        self.x_init = np.zeros(2)
        self.x_init[:] = x_init
        self.x_target = np.zeros(2)
        self.x_target[:] = x_target
        self.coords = coords
        # self.aero = LLAero(mode='dabry')
        self.aero = MermozAero()

        # Domain bounding box corners
        if bl is not None:
            self.bl = np.zeros(2)
            self.bl[:] = bl
        else:
            self.bl = None
        if tr is not None:
            self.tr = np.zeros(2)
            self.tr[:] = tr
        else:
            self.tr = None

        self.geod_l = Utils.distance(self.x_init, self.x_target, coords=self.coords)

        # It is usually sufficient to scale time on geodesic / airspeed
        # but for some problems it isn't
        self.time_scale = self.geod_l / self.model.v_a if time_scale is None else time_scale

        self.bm = None

        self.descr = 'Problem' if descr is None else descr

        if not domain:
            if not autodomain:
                if self.bl is not None and self.tr is not None:
                    self.domain = lambda x: self.bl[0] < x[0] < self.tr[0] and self.bl[1] < x[1] < self.tr[1]
                    self.geod_l = self.distance(self.bl, self.tr)
                else:
                    self.domain = lambda _: True
            else:
                # Bound computation domain on wind grid limits
                wind = self.model.wind
                if type(wind) == DiscreteWind:
                    self.bl = np.array((wind.x_min, wind.y_min))
                    self.tr = np.array((wind.x_max, wind.y_max))
                else:
                    w = 1.15 * self.geod_l
                    self.bl = (self.x_init + self.x_target) / 2. - np.array((w / 2., w / 2.))
                    self.tr = (self.x_init + self.x_target) / 2. + np.array((w / 2., w / 2.))
                if self.coords == Utils.COORD_GCS:
                    factor = 1. if wind.units_grid == Utils.U_DEG else Utils.RAD_TO_DEG
                self.domain = lambda x: self.bl[0] < x[0] < self.tr[0] and self.bl[1] < x[1] < self.tr[1]
        else:
            if self.bl is not None and self.tr is not None:
                self.domain = lambda x: self.bl[0] < x[0] < self.tr[0] and self.bl[1] < x[1] < self.tr[1] and domain(x)
            else:
                self.domain = domain

        if self.bl is not None and self.tr is not None:
            self.l_ref = self.distance(self.bl, self.tr)
        else:
            self.l_ref = self.geod_l

        self.obstacles = obstacles if obstacles is not None else []
        if len(self.obstacles) > 0:
            for obs in self.obstacles:
                obs.update_lref(self.l_ref)

        self.frame_offset = 0.05
        if autoframe:
            bl_frame = self.bl + (self.tr - self.bl) * self.frame_offset / 2.
            tr_frame = self.tr - (self.tr - self.bl) * self.frame_offset / 2.
            self.obstacles.append(FrameObs(bl_frame, tr_frame))

        self._feedback = None
        self._mfb = None
        self._aslaw = None
        self.trajs = []

    def update_airspeed(self, v_a):
        self.model.update_airspeed(v_a)

    def __str__(self):
        return str(self.descr)

    def load_feedback(self, feedback: Feedback):
        """
        Load a feedback law for integration

        :param feedback: A feedback law
        """
        self._feedback = feedback

    def load_aslaw(self, aslaw: AirspeedLaw):
        self._aslaw = aslaw

    def load_multifb(self, feedback: MultiFeedback):
        self._mfb = feedback

    def integrate_trajectory(self,
                             x_init: ndarray,
                             stop_cond: StoppingCond,
                             max_iter=20000,
                             int_step=0.0001,
                             t_init=0.,
                             backward=False):
        """
        Use the specified discrete integration method to build a trajectory
        with the given control law. Store the integrated trajectory in
        the trajectory list.
        :param x_init: The initial state
        :param stop_cond: A stopping condition for the integration
        :param max_iter: Maximum number of iterations
        :param int_step: Integration step
        :param t_init: Initial timestamp
        """
        sc = TimedSC(1.)
        sc.value = lambda t, x: stop_cond.value(t, x) or not self.domain(x)
        integrator = IntEulerExpl(self.model.wind,
                                  self.model.dyn,
                                  self._feedback if self._feedback is not None else self._mfb,
                                  self.coords,
                                  aslaw=self._aslaw,
                                  stop_cond=sc,
                                  max_iter=max_iter,
                                  int_step=int_step,
                                  t_init=t_init,
                                  backward=backward)
        traj = integrator.integrate(x_init)
        self.trajs.append(traj)
        return traj

    def dualize(self):
        # Create the dual problem, i.e. going from target to init in the mirror wind (multiplication by -1)
        dual_model = ZermeloGeneralModel(self.model.v_a, self.coords)
        dual_model.update_wind(self.model.wind.dualize())

        kwargs = {}
        for k, v in self.__dict__.items():
            if k == 'model':
                kwargs[k] = dual_model
            elif k == 'x_init':
                kwargs[k] = np.zeros(2)
                kwargs[k][:] = self.x_target
            elif k == 'x_target':
                kwargs[k] = np.zeros(2)
                kwargs[k][:] = self.x_init
            else:
                kwargs[k] = v

        return NavigationProblem(**kwargs)

    def flatten(self):
        if self.coords != Utils.COORD_GCS:
            raise Exception('Flattening only available in GCS mode')
        self.coords = Utils.COORD_CARTESIAN
        new_model = ZermeloGeneralModel(self.model.v_a, coords=Utils.COORD_CARTESIAN)
        lon_0, lat_0 = Utils.middle(self.x_init, self.x_target, coords=Utils.COORD_GCS)
        self.model.wind.flatten(proj='ortho', lon_0=lon_0, lat_0=lat_0)
        proj = Proj(proj='ortho', lon_0=Utils.RAD_TO_DEG * lon_0, lat_0=Utils.RAD_TO_DEG * lat_0)
        self.x_init = np.array(proj(*(Utils.RAD_TO_DEG * self.x_init)))
        self.x_target = np.array(proj(*(Utils.RAD_TO_DEG * self.x_target)))
        self.bl = np.array(proj(*(Utils.RAD_TO_DEG * self.bl)))
        self.tr = np.array(proj(*(Utils.RAD_TO_DEG * self.tr)))
        new_model.update_wind(self.model.wind)
        self.model = new_model
        self.geod_l = self.distance(self.x_init, self.x_target)
        self.l_ref = self.distance(self.bl, self.tr)
        self.domain = lambda x: self.bl[0] < x[0] < self.tr[0] and self.bl[1] < x[1] < self.tr[1]

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

    def eliminate_trajs(self, target, tol: float):
        """
        Delete trajectories that are too far from the objective point
        :param tol: The radius tolerance around the objective in meters
        """
        delete_index = []
        for k, traj in enumerate(self.trajs):
            keep = False
            for id, p in enumerate(traj.points):
                if np.linalg.norm(p - target) < tol:
                    keep = True
                    break
            if not keep:
                delete_index.append(k)
        for index in sorted(delete_index, reverse=True):
            del self.trajs[index]

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
        fb = GSTargetFB(self.model.wind, self.model.v_a, self.x_target, self.coords)
        self.load_feedback(fb)
        sc = DistanceSC(lambda x: self.distance(x, self.x_target), self.geod_l * 0.02)
        traj = self.integrate_trajectory(self.x_init, sc, int_step=self.time_scale/1000, max_iter=3000)
        if sc.value(0, traj.points[traj.last_index]):
            return traj.timestamps[traj.last_index]
        else:
            return -1


class DatabaseProblem(NavigationProblem):

    def __init__(self, wind_fpath, x_init=None, x_target=None, airspeed=Utils.AIRSPEED_DEFAULT, obstacles=None):
        total_wind = DiscreteWind(interp='linear')
        total_wind.load(wind_fpath)
        print(f'Problem from database : {wind_fpath}')

        with h5py.File(wind_fpath, 'r') as f:
            coords = f.attrs['coords']
            bl = np.array((f['grid'][0, 0, 0], f['grid'][0, 0, 1]))
            tr = np.array((f['grid'][-1, -1, 0], f['grid'][-1, -1, 1]))
            if x_init is None:
                print('Automatic parameters')
                offset = (tr - bl) * 0.1
                x_init = bl + offset
                x_target = tr - offset

        if obstacles is None:
            obstacles = []

        offset = (tr - bl) * 0.02
        bl_obs = bl + offset
        tr_obs = tr - offset

        obstacles.append(MeridianObs(bl_obs[0], True))
        obstacles.append(MeridianObs(tr_obs[0], False))
        obstacles.append(ParallelObs(bl_obs[1], True))
        obstacles.append(ParallelObs(tr_obs[1], False))
        eps_lon = (tr_obs[0] - bl_obs[0]) / 30
        eps_lat = (tr_obs[1] - bl_obs[1]) / 30
        obstacles.append(GreatCircleObs(np.array((bl_obs[0], bl_obs[1])) + np.array((eps_lon, 0)),
                                        np.array((bl_obs[0], bl_obs[1])) + np.array((0, eps_lat))))
        obstacles.append(GreatCircleObs(np.array((tr_obs[0], bl_obs[1])) + np.array((0, eps_lat)),
                                        np.array((tr_obs[0], bl_obs[1])) + np.array((-eps_lon, 0))))
        obstacles.append(GreatCircleObs(np.array((tr_obs[0], tr_obs[1])) + np.array((-eps_lon, 0)),
                                        np.array((tr_obs[0], tr_obs[1])) + np.array((0, -eps_lat))))
        obstacles.append(GreatCircleObs(np.array((bl_obs[0], tr_obs[1])) + np.array((0, -eps_lat)),
                                        np.array((bl_obs[0], tr_obs[1])) + np.array((eps_lon, 0))))

        zermelo_model = ZermeloGeneralModel(airspeed, coords=coords)
        zermelo_model.update_wind(total_wind)
        super().__init__(zermelo_model, x_init, x_target, coords, obstacles=obstacles, mask_land=False)


class IndexedProblem(NavigationProblem):
    problems = {
        0: ['Three vortices', '3vor'],
        1: ['Linear wind', 'linear'],
        2: ['Honolulu Vancouver', 'honolulu-vancouver'],
        3: ['Double Gyre Li2020', 'double-gyre-li2020'],
        4: ['Double Gyre Kularatne2016', 'double-gyre-ku2016'],
        5: ['Point symmetric Techy2011', 'pointsym-techy2011'],
        6: ['Three obstacles', '3obs'],
        7: ['San-Juan Dublin Ortho', 'sanjuan-dublin-ortho'],
        8: ['Big Rankine vortex', 'big_rankine'],
        9: ['Four vortices', '4vor'],
        10: ['Moving_vortex', 'movor'],
        11: ['One obstacle', '1obs'],
        12: ['Moving obstacle', 'movobs'],
        13: ['Time-varying linear wind', 'tvlinear'],
        14: ['Moving vortices', 'movors'],
        15: ['Gyre Rhoads2010', 'gyre-rhoads2010'],
        16: ['Gyre Transversality', 'gyre-transver'],
        17: ['Band wind', 'band'],
        18: ['Linearly varying wind', 'lva'],
        19: ['Double gyre scaled', 'double-gyre-scaled'],
        20: ['Trap wind', 'trap'],
        21: ['San Juan Dublin Flattened Time varying', 'sanjuan-dublin-ortho-tv'],
        22: ['Obstacle', 'obs'],
        23: ['Chertovskih', 'chertov']
    }
    exclude_from_test = [12]

    def __init__(self, i, seed=0):
        if i == 0:
            v_a = 14.11
            coords = Utils.COORD_CARTESIAN

            f = 1e6
            fs = 15.46

            x_init = f * np.array([0., 0.])
            x_target = f * np.array([1., 0.])

            bl = f * np.array([-2, -2])
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

            vortex1 = RankineVortexWind(omega1, f * fs * -1., f * 1e-1)
            vortex2 = RankineVortexWind(omega2, f * fs * -0.8, f * 1e-1)
            vortex3 = RankineVortexWind(omega3, f * fs * 0.8, f * 1e-1)
            const_wind = UniformWind(np.array([0., 0.]))

            alty_wind = 3. * const_wind + vortex1 + vortex2 + vortex3
            total_wind = LCWind(np.array((3., 1., 1., 1.)),
                                (const_wind, vortex1, vortex2, vortex3))
            # total_wind = DiscreteWind()
            # total_wind.load_from_wind(alty_wind, 101, 101, bl, tr, coords=coords)

            zermelo_model = ZermeloGeneralModel(v_a)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords)

        elif i == 1:
            v_a = 23.
            coords = Utils.COORD_CARTESIAN

            f = 1e6
            x_init = f * np.array([0., 0.])
            x_target = f * np.array([1., 0.])

            gradient = np.array([[0., v_a / f], [0., 0.]])
            origin = np.array([0., 0.])
            value_origin = np.array([0., 0.])

            bl = f * np.array([-0.2, -1.])
            tr = f * np.array([1.2, 1.])

            linear_wind = LinearWind(gradient, origin, value_origin)
            # total_wind = DiscreteWind()
            # total_wind.load_from_wind(linear_wind, 51, 51, bl, tr, coords)

            # Creates the cinematic model
            v_a = 11.41
            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(linear_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr)

        elif i == 2:
            v_a = 23.
            coords = Utils.COORD_GCS

            total_wind = DiscreteWind(interp='pwc')
            total_wind.load('/home/bastien/Documents/data/wind/windy/Vancouver-Honolulu-0.5.mz/data2.h5')

            # Creates the cinematic model
            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            # Get problem domain boundaries
            bl = np.zeros(2)
            tr = np.zeros(2)
            bl[:] = total_wind.grid[1, 1]
            tr[:] = total_wind.grid[-2, -2]

            # Initial point
            offset = np.array([5., 5.])  # Degrees
            x_init = Utils.DEG_TO_RAD * (np.array([-157.855676, 21.304547]) + offset)
            x_target = Utils.DEG_TO_RAD * (np.array([-123.113952, 49.2608724]) - offset)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords)

        elif i == 3:
            v_a = 0.6

            sf = 1.

            x_init = sf * np.array((125., 125.))
            x_target = sf * np.array((375., 375.))
            bl = sf * np.array((-10, -10))
            tr = sf * np.array((510, 510))
            coords = Utils.COORD_CARTESIAN

            total_wind = DoubleGyreWind(0., 0., 500., 500., 1.)

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr)

        elif i == 4:
            v_a = 0.05

            sf = 1.

            x_init = sf * np.array((0.6, 0.6))
            x_target = sf * np.array((2.4, 2.4))
            bl = sf * np.array((0.5, 0.5))
            tr = sf * np.array((2.5, 2.5))
            coords = Utils.COORD_CARTESIAN

            total_wind = DoubleGyreWind(0.5, 0.5, 2., 2., pi * 0.02)

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr)

        elif i == 5:
            v_a = 18.
            sf = 1e6

            x_init = sf * np.array((0., 0.))
            x_target = sf * np.array((1., 0.))
            bl = sf * np.array((-0.1, -1.))
            tr = sf * np.array((1.1, 1.))
            coords = Utils.COORD_CARTESIAN

            # To get w as wind value at start point, choose gamma = w / 0.583
            gamma = v_a / sf * 1.
            omega = 0.
            total_wind = PointSymWind(sf * 0.5, sf * 0.3, gamma, omega)

            v_a = 14.66
            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr)

        elif i == 6:

            v_a = 23.

            sf = 3e6

            x_init = sf * np.array((0., 0.))
            x_target = sf * np.array((1., 0.))
            bl = sf * np.array((-0.15, -1.15))
            tr = sf * np.array((1.15, 1.15))
            coords = Utils.COORD_CARTESIAN

            const_wind = UniformWind(np.array([1., 1.]))

            c1 = sf * np.array((0.5, 0.))
            c2 = sf * np.array((0.5, 0.1))
            c3 = sf * np.array((0.5, -0.1))

            obstacle1 = RadialGaussWind(c1[0], c1[1], sf * 0.1, 1 / 2 * 0.2, v_a * 5.)
            obstacle2 = RadialGaussWind(c2[0], c2[1], sf * 0.1, 1 / 2 * 0.2, v_a * 5.)
            obstacle3 = RadialGaussWind(c3[0], c3[1], sf * 0.1, 1 / 2 * 0.2, v_a * 5.)

            alty_wind = obstacle1 + obstacle2 + obstacle3 + const_wind

            total_wind = alty_wind  # DiscreteWind()

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            obs_center = [c1, c2, c3]
            obs_radius = sf * np.array((0.13, 0.13, 0.13))
            phi_obs = {}
            for j in range(obs_radius.shape[0]):
                def f(t, x, c=obs_center[j], r=obs_radius[j]):
                    return (x - c) @ np.diag((1., 1.)) @ (x - c) - r ** 2

                phi_obs[j] = f

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, phi_obs=phi_obs, bl=bl, tr=tr)

        elif i == 7:
            v_a = 23.
            coords = Utils.COORD_CARTESIAN
            total_wind = DiscreteWind()
            total_wind.load('/home/bastien/Documents/data/wind/ncdc/san-juan-dublin-flattened-ortho.mz/wind.h5')

            bl = total_wind.grid[0, 0]
            tr = total_wind.grid[-1, -1]

            proj = Proj(proj='ortho', lon_0=total_wind.lon_0, lat_0=total_wind.lat_0)

            point = np.array([-66.116666, 18.465299])
            x_init = np.array(proj(*point)) + 500e3 * np.ones(2)

            point = np.array([-6.2602732, 53.3497645])
            x_target = np.array(proj(*point)) - 500e3 * np.ones(2)

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(-1. * total_wind)

            # Creates the navigation problem on top of the previous model
            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr)

        elif i == 8:
            v_a = 23.
            coords = Utils.COORD_CARTESIAN

            f = 1e6
            fs = v_a

            x_init = f * np.array([0., 0.])
            x_target = f * np.array([1., 0.])

            bl = f * np.array([-0.2, -1.])
            tr = f * np.array([1.2, 1.])
            omega = f * np.array(((0.2, -0.2), (0.8, 0.2)))

            vortex = [
                RankineVortexWind(omega[0], f * fs * -7., f * 1.),
                RankineVortexWind(omega[1], f * fs * -7., f * 1.)
            ]
            const_wind = UniformWind(np.array([0., 0.]))

            alty_wind = const_wind + vortex[0] + vortex[1]
            # total_wind = DiscreteWind()
            # total_wind.load_from_wind(alty_wind, 101, 101, bl, tr, coords=coords)

            zermelo_model = ZermeloGeneralModel(v_a)
            zermelo_model.update_wind(alty_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr)

        elif i == 9:
            v_a = 23.
            coords = Utils.COORD_CARTESIAN

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
            vortices = [RankineVortexWind(omega[i], strength[i], radius[i]) for i in range(len(omega))]
            const_wind = UniformWind(np.array([0., 0.]))

            alty_wind = sum(vortices, const_wind)
            # total_wind = DiscreteWind()
            # total_wind.load_from_wind(alty_wind, 101, 101, bl, tr, coords=coords)

            zermelo_model = ZermeloGeneralModel(v_a)
            zermelo_model.update_wind(alty_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr)

        elif i == 10:
            v_a = 23.
            coords = Utils.COORD_CARTESIAN

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

            vortex = RankineVortexWind(omega, gamma, radius, t_end=1. * f / v_a)
            const_wind = UniformWind(np.array([0., 5.]))

            total_wind = vortex

            zermelo_model = ZermeloGeneralModel(v_a)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords)

        elif i == 11:

            v_a = 23.

            sf = 3e6

            x_init = sf * np.array((0., 0.))
            x_target = sf * np.array((1., 0.))
            bl = sf * np.array((-0.15, -1.15))
            tr = sf * np.array((1.15, 1.15))
            coords = Utils.COORD_CARTESIAN

            obs_center = [
                sf * np.array((0.15, 0.1)),
                sf * np.array((0.5, -0.2)),
                sf * np.array((0.8, 0.1))
            ]
            obs_radius = sf * np.array((0.15, 0.15, 0.15))

            const_wind = UniformWind(np.array([1., 0.]))
            band = BandGaussWind(sf * np.array((0., -.35)),
                                 sf * np.array((1., 0.)),
                                 23.,
                                 sf * 0.05)

            if seed:
                obs_radius *= 0.85
                total_wind = LCWind(
                    np.ones(4),
                    (const_wind,
                     RadialGaussWind(
                         obs_center[0][0],
                         obs_center[0][1],
                         obs_radius[0],
                         1 / 2 * 0.2,
                         v_a * 5.),
                     RadialGaussWind(
                         obs_center[1][0],
                         obs_center[1][1],
                         obs_radius[1],
                         1 / 2 * 0.2,
                         v_a * 5.),
                     RadialGaussWind(
                         obs_center[2][0],
                         obs_center[2][1],
                         obs_radius[2],
                         1 / 2 * 0.2,
                         v_a * 5.))
                )
                obs_radius *= 1.1 / 0.9
            else:
                total_wind = const_wind + band
            phi_obs = {}
            for k in range(obs_radius.shape[0]):
                def f(t, x, c=obs_center[k], r=obs_radius[k]):
                    return (x - c) @ np.diag((1., 1.)) @ (x - c) - r ** 2

                phi_obs[k] = f

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            # c = sf * np.array((0.5, 0.2))
            # r = sf * 0.1
            # c2 = sf * np.array((0.5, -0.2))
            # r2 = sf * 0.05
            # phi_obs = {}
            # phi_obs[0] = lambda x: (x - c) @ np.diag((1., 1.)) @ (x - c) - r ** 2
            # phi_obs[1] = lambda x: (x - c2) @ np.diag((2., 1.)) @ (x - c2) - r2 ** 2

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr,
                                                 phi_obs=phi_obs)
        elif i == 12:

            v_a = 23.

            sf = 3e6

            x_init = sf * np.array((0., 0.))
            x_target = sf * np.array((1., 0.))
            bl = sf * np.array((-0.15, -1.15))
            tr = sf * np.array((1.15, 1.15))
            coords = Utils.COORD_CARTESIAN

            nt = 20

            obs_x = sf * np.linspace(0.5, 0.5, nt)
            obs_y = sf * np.linspace(-0.2, 0.2, nt)
            obs_center = np.column_stack((obs_x, obs_y))
            obs_radius = sf * np.linspace(0.15, 0.15, nt)
            obs_sdev = np.linspace(1 / 2 * 0.2, 1 / 2 * 0.2, nt)
            obs_v_max = np.linspace(v_a * 5., v_a * 5., nt)

            t_end = 0.8 * Utils.distance(x_init, x_target, coords) / v_a

            total_wind = LCWind(
                np.ones(2),
                (UniformWind(np.array((5., -5.))),
                 RadialGaussWindT(obs_center, obs_radius, obs_sdev, obs_v_max, t_end))
            )

            # phi_obs = {}
            # for k in range(obs_radius.shape[0]):
            #     def f(x, c=obs_center[k], r=obs_radius[k]):
            #         return (x - c) @ np.diag((1., 1.)) @ (x - c) - r ** 2
            #
            #     phi_obs[k] = f

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            phi_obs = {}

            def f(t, x):
                tt = t / t_end
                k, alpha = None, None
                if tt > 1.:
                    k, alpha = nt - 2, 1.
                elif tt < 0.:
                    k, alpha = 0, 0.
                else:
                    k = int(tt * (nt - 1))
                    if k == nt - 1:
                        k = nt - 2
                        alpha = 1.
                    else:
                        alpha = tt * (nt - 1) - k
                c = (1 - alpha) * obs_center[k] + alpha * obs_center[k + 1]
                r = (1 - alpha) * obs_radius[k] + alpha * obs_radius[k + 1]
                return (x - c) @ np.diag((1., 1.)) @ (x - c) - r ** 2

            phi_obs[0] = f

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr, phi_obs=phi_obs)
            # phi_obs=phi_obs)

        elif i == 13:
            v_a = 23.
            coords = Utils.COORD_CARTESIAN

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

            linear_wind = LinearWindT(gradient, origin, value_origin, t_end=0.5 * f / v_a)
            # total_wind = DiscreteWind()
            # total_wind.load_from_wind(linear_wind, 51, 51, bl, tr, coords)

            # Creates the cinematic model
            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(linear_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr)
        elif i == 14:
            v_a = 23.
            coords = Utils.COORD_CARTESIAN

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

                vortices.append(RankineVortexWind(omega, gamma, radius, t_end=1. * f / v_a))

            obstacles = []
            # obstacles.append(RadialGaussWind(f * 0.5, f * 0.15, f * 0.1, 1 / 2 * 0.2, 10*v_a))
            winds = [UniformWind(np.array((-5., 0.)))] + vortices + obstacles
            # const_wind = UniformWind(np.array([0., 5.]))
            N = len(winds) - len(obstacles)
            M = len(obstacles)

            total_wind = LCWind(np.array(tuple(1 / N for _ in range(N)) + tuple(1. for _ in range(M))), tuple(winds))

            zermelo_model = ZermeloGeneralModel(v_a)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords)
        elif i == 15:
            v_a = 23.

            sf = 3e6

            x_init = sf * np.array((0, 0.25))
            x_target = sf * np.array((0.03, -0.25))
            bl = sf * np.array((-1, -0.5))
            tr = sf * np.array((1., 0.5))
            coords = Utils.COORD_CARTESIAN

            total_wind = DoubleGyreWind(sf * 0, sf * -0.5, sf * 2, sf * 2, 30.)

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr,
                                                 autodomain=False)
            self.time_scale = 3. * self.geod_l / v_a
        elif i == 16:
            v_a = 0.6

            sf = 1.

            x_init = sf * np.array((250., 0.))
            x_target = sf * np.array((250., 125.))
            bl = sf * np.array((-10, -100))
            tr = sf * np.array((460, 250))
            coords = Utils.COORD_CARTESIAN

            total_wind = DoubleGyreWind(0., 0., 500., 500., 1.)

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr,
                                                 autodomain=False)
            self.time_scale = 3. * self.geod_l / v_a
        elif i == 17:
            v_a = 20.7

            sf = 1.

            x_init = sf * np.array((20., 20.))
            x_target = sf * np.array((80., 80.))
            bl = sf * np.array((15, 15))
            tr = sf * np.array((85, 85))
            coords = Utils.COORD_CARTESIAN

            band_wind = BandWind(np.array((0., 50.)), np.array((1., 0.)), np.array((-20., 0.)), 20)
            total_wind = DiscreteWind()
            total_wind.load_from_wind(band_wind, 101, 101, bl, tr, coords, fd=True)

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr,
                                                 autodomain=False)
            self.time_scale = 1. * self.geod_l / v_a
        elif i == 18:
            v_a = 23.

            sf = 3e6

            x_init = sf * np.array((0.1, 0.))
            x_target = sf * np.array((0.9, 0.))
            coords = Utils.COORD_CARTESIAN
            wind_value = np.array((0., -15.))
            gradient = np.array((0., 30 / 130000))
            time_scale = 130000
            lv_wind = LVWind(wind_value, gradient, time_scale)

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(lv_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords)
            self.time_scale = time_scale

        elif i == 19:
            v_a = 15.

            sf = 3e6

            x_init = sf * np.array((0.6, 0.6))
            x_target = sf * np.array((2.4, 2.4))
            bl = sf * np.array((0.5, 0.5))
            tr = sf * np.array((2.5, 2.5))
            coords = Utils.COORD_CARTESIAN

            total_wind = DoubleGyreWind(sf * 0.5, sf * 0.5, sf * 2., sf * 2., v_a / 2 * pi)

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr, autoframe=True)

        elif i == 20:
            v_a = 23.

            sf = 3e6

            x_init = sf * np.array((0., 0.))
            x_target = sf * np.array((1., 0.))
            bl = sf * np.array((-0.2, -0.6))
            tr = sf * np.array((1.2, 0.6))
            coords = Utils.COORD_CARTESIAN

            nt = 40
            wind_value = 80 * np.ones(nt)
            center = np.zeros((nt, 2))
            for k in range(30):
                center[10 + k] = sf * np.array((0.05 * k, 0.))
            radius = sf * 0.2 * np.ones(nt)

            total_wind = TrapWind(wind_value, center, radius, t_end=400000)

            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(total_wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr,
                                                 autodomain=False)

        elif i == 21:
            v_a = 23.

            x_init = np.array((1.6e6, 1.6e6))
            x_target = np.array((-2e6, 0.5e6))
            bl = np.array((-2e6, -1.5e6))
            tr = np.array((2e6, 2e6))
            coords = Utils.COORD_CARTESIAN
            wind = DiscreteWind()
            wind.load('/home/bastien/Documents/data/wind/ncdc/san-juan-dublin-flattened-ortho-tv.mz/wind.h5')
            zermelo_model = ZermeloGeneralModel(v_a, coords=coords)
            zermelo_model.update_wind(wind)

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, bl=bl, tr=tr,
                                                 autodomain=False)

        elif i == 22:
            v_a = 23.

            sf = 3e6
            x_init = sf * np.array((0.1, 0.))
            x_target = sf * np.array((0.9, 0.))
            wind = UniformWind(np.array((5., 5.)))
            coords = Utils.COORD_CARTESIAN
            zermelo_model = ZermeloGeneralModel(v_a, coords)
            zermelo_model.update_wind(wind)

            obstacles = []
            # obstacles.append(CircleObs(sf * np.array((0.4, 0.1)), sf * 0.1))
            # obstacles.append(CircleObs(sf * np.array((0.5, 0.)), sf * 0.1))
            obstacles.append(MeanObs([CircleObs(sf * np.array((0.4, 0.1)), sf * 0.1),
                                         CircleObs(sf * np.array((0.5, 0.)), sf * 0.1)]))
            obstacles.append(FrameObs(sf * np.array((0.05, -0.45)), sf * np.array((0.95, 0.45))))

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, obstacles=obstacles)
        elif i == 23:
            v_a = 1
            x_init = np.array((0.5, 0))
            x_target = np.array((-0.7, -6))
            coords = Utils.COORD_CARTESIAN
            wind = ChertovskihWind()
            zermelo_model = ZermeloGeneralModel(v_a, coords)
            zermelo_model.update_wind(wind)

            obstacles = []
            obstacles.append(FrameObs(np.array((-1, -6.01)), np.array((1, 0.01))))

            super(IndexedProblem, self).__init__(zermelo_model, x_init, x_target, coords, obstacles=obstacles,
                                                 bl=np.array((-1.5, -6.05)),
                                                 tr=np.array((1.5, 0.05)), autodomain=False)


        else:
            raise IndexError(f'No problem with index {i}')

        self.descr = IndexedProblem.problems[i][0]