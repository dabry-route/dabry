import json
import os
import sys
from math import atan2

import h5py
import numpy as np
import pyproj
import scipy.io
from matplotlib import pyplot as plt
from numpy import cos, sin

from dabry.misc import Utils
from dabry.problem import NavigationProblem
from dabry.trajectory import Trajectory
from dabry.wind import DiscreteWind

"""
rft.py
Provides basic function to carry out reachability front tracking
on navigation problems.

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


def upwind_diff(field, axis, delta):
    """
    Computes the derivative in the given direction with the given discretization step
    using an upwind scheme
    :param field: The field which derivative must be computed
    :param axis: The direction in which to compute the derivative
    :param delta: The discretization step
    :return: The derivated field
    """
    if axis not in [0, 1]:
        print("Only handles 2D fields")
        exit(1)
    nx, ny = field.shape
    res = np.zeros(field.shape)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if axis == 0:
                if field[i - 1, j] < field[i, j] < field[i + 1, j]:
                    res[i, j] = (field[i, j] - field[i - 1, j]) / delta
                elif field[i - 1, j] > field[i, j] > field[i + 1, j]:
                    res[i, j] = (field[i + 1, j] - field[i, j]) / delta
                elif field[i - 1, j] <= field[i, j] >= field[i + 1, j]:
                    res[i, j] = (field[i + 1, j] - field[i - 1, j]) / (2 * delta)
                elif field[i - 1, j] >= field[i, j] <= field[i + 1, j]:
                    res[i, j] = 0
            else:
                # axis == 1
                if field[i, j - 1] < field[i, j] < field[i, j + 1]:
                    res[i, j] = (field[i, j] - field[i, j - 1]) / delta
                elif field[i, j - 1] > field[i, j] > field[i, j + 1]:
                    res[i, j] = (field[i, j + 1] - field[i, j]) / delta
                elif field[i, j - 1] <= field[i, j] >= field[i, j + 1]:
                    res[i, j] = (field[i, j + 1] - field[i, j - 1]) / (2 * delta)
                elif field[i, j - 1] >= field[i, j] <= field[i, j + 1]:
                    res[i, j] = 0
    for j in range(1, field.shape[1] - 1):
        res[0, j] = res[1, j]
        res[nx - 1, j] = res[nx - 2, j]
    for i in range(1, field.shape[0] - 1):
        res[i, 0] = res[i, 1]
        res[i, ny - 1] = res[i, ny - 2]
    res[0, 0] = res[1, 1]
    res[nx - 1, 0] = res[nx - 2, 1]
    res[0, ny - 1] = res[1, ny - 2]
    res[nx - 1, ny - 1] = res[nx - 2, ny - 2]
    return res


def wind_dot_phi(field, windfield, delta):
    """
    Computes the advection term "wind dot field" with the given discretization step
    using an upwind scheme
    From Sethian 6.45
    :param field: The field which derivative must be computed
    :param windfield: The windfield advecting the previous quantity
    :param delta: The discretization step
    :return: The gradient of the field
    """
    nx, ny = field.shape
    res = np.zeros(field.shape)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            d_mx = (field[i, j] - field[i - 1, j]) / delta
            d_px = (field[i + 1, j] - field[i, j]) / delta
            d_my = (field[i, j] - field[i, j - 1]) / delta
            d_py = (field[i, j + 1] - field[i, j]) / delta
            u_ijn = windfield[i, j, 0]
            v_ijn = windfield[i, j, 1]
            res[i, j] = max(u_ijn, 0.) * d_mx + min(u_ijn, 0.) * d_px + max(v_ijn, 0.) * d_my + min(v_ijn, 0.) * d_py

    for j in range(1, field.shape[1] - 1):
        res[0, j] = res[1, j]
        res[nx - 1, j] = res[nx - 2, j]
    for i in range(1, field.shape[0] - 1):
        res[i, 0] = res[i, 1]
        res[i, ny - 1] = res[i, ny - 2]
    res[0, 0] = res[1, 1]
    res[nx - 1, 0] = res[nx - 2, 1]
    res[0, ny - 1] = res[1, ny - 2]
    res[nx - 1, ny - 1] = res[nx - 2, ny - 2]
    return res


def central_diff(field, axis, delta):
    """
    Computes the derivative in the given direction with the given discretization step
    using a central difference scheme
    :param field: The field which derivative must be computed
    :param axis: The direction in which to compute the derivative
    :param delta: The discretization step
    :return: The derivated field
    """
    if axis not in [0, 1]:
        print("Only handles 2D fields")
        exit(1)
    nx, ny = field.shape
    res = np.zeros(field.shape)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if axis == 0:
                res[i, j] = (field[i + 1, j] - field[i - 1, j]) / (2 * delta)
            else:
                # axis == 1
                res[i, j] = (field[i, j + 1] - field[i, j - 1]) / (2 * delta)
    for j in range(1, field.shape[1] - 1):
        res[0, j] = res[1, j]
        res[nx - 1, j] = res[nx - 2, j]
    for i in range(1, field.shape[0] - 1):
        res[i, 0] = res[i, 1]
        res[i, ny - 1] = res[i, ny - 2]
    res[0, 0] = res[1, 1]
    res[nx - 1, 0] = res[nx - 2, 1]
    res[0, ny - 1] = res[1, ny - 2]
    res[nx - 1, ny - 1] = res[nx - 2, ny - 2]
    return res


def grad(field, delta_x, delta_y, method='central'):
    """
    Computes the gradient of a field in a
    :param field: The field which gradient should be computed
    :param delta_x: The discretization step in the x-axis direction
    :param delta_y: The discretization step in the y-axis direction
    :param method: 'central' or 'upwind'
    :return: The gradient of the field
    """
    if method == 'central':
        diff0 = central_diff(field, 0, delta_x)
        diff1 = central_diff(field, 1, delta_y)
    elif method == 'upwind':
        diff0 = upwind_diff(field, 0, delta_x)
        diff1 = upwind_diff(field, 1, delta_y)
    else:
        print(f'Unknown method :{method}')
        exit(1)

    return np.stack((diff0, diff1), axis=2)


def norm_grad(field, delta_x, delta_y, method='central'):
    """
    Computes the norm of a field's gradient with the given method
    :param field: The field
    :param delta_x: The discretization step in the x-axis direction
    :param delta_y: The discretization step in the y-axis direction
    :param method: 'central' or 'upwind'
    :return: The norm of the gradient
    """
    if method == 'central':
        diff0 = central_diff(field, 0, delta_x)
        diff1 = central_diff(field, 1, delta_y)
    elif method == 'upwind':
        diff0 = upwind_diff(field, 0, delta_x)
        diff1 = upwind_diff(field, 1, delta_y)
    else:
        print(f'Unknown method :{method}')
        exit(1)
    return np.sqrt(diff0 ** 2 + diff1 ** 2)


class RFT:
    """
    Python interface for reachability front tracker
    """

    def __init__(self, bl, tr, max_time, nx, ny, nt, mp: NavigationProblem, kernel='matlab', method='sethian'):
        """
        :param bl: Grid bottom left point
        :param tr: Grid top right point
        :param max_time: Time window upper bound
        :param nx: Number of points in the x-axis direction
        :param ny: Number of points in the y-axis direction
        :param nt: Number of points in the time direction.
        :param mp: The dabry problem instance containing the analytic wind
        :param kernel: 'base' for Python implementation of level set methods or 'matlab'
        to use ToolboxLS (https://www.cs.ubc.ca/~mitchell/ToolboxLS)
        :param method: With 'base' kernel, select numerical scheme ('sethian' or 'lolla')
        """

        self.bl = np.zeros(2)
        self.bl[:] = bl
        self.tr = np.zeros(2)
        self.tr[:] = tr
        self.max_time = max_time

        self.nx = nx
        self.ny = ny
        self.nt = nt

        if self.nt is not None:
            self.delta_t = self.max_time / (self.nt - 1)
        else:
            self.delta_t = None

        self.mp = mp
        self.proj = None
        self.x_init = np.zeros(2)
        self.x_init[:] = self.mp.x_init

        if self.mp.coords == Utils.COORD_GCS:
            # When working in spherical coordinates, we will first flatten the
            # problem using an orthographic projection
            mid = self.mp.middle(self.mp.x_init, self.mp.x_target)
            self.lon_0, self.lat_0 = Utils.RAD_TO_DEG * mid[0], Utils.RAD_TO_DEG * mid[1]
            self.proj = pyproj.Proj(proj='ortho', lon_0=self.lon_0, lat_0=self.lat_0)
            # self.proj = pyproj.Proj(proj='omerc',
            #             lon_1=self.mp.x_init[0],
            #             lat_1=self.mp.x_init[1],
            #             lon_2=self.mp.x_target[0],
            #             lat_2=self.mp.x_target[1])
            # New boundaries are defined as an encapsulating bounding box over the region defined
            # by (lon, lat) bottom left and top right corners
            # Define a dense grid of points within the (lon, lat) zone
            x = Utils.RAD_TO_DEG * self.mp.model.wind.grid[:, :, 0]
            y = Utils.RAD_TO_DEG * self.mp.model.wind.grid[:, :, 1]
            # x, y = np.meshgrid(RAD_TO_DEG * np.linspace(self.bl[0], self.tr[0], 100),
            #                    RAD_TO_DEG * np.linspace(self.bl[1], self.tr[1], 100), indexing='ij')
            # Project it
            x, y = self.proj(x, y)

            # Find encapsulating box
            self.bl = np.array((np.min(x), np.min(y)))
            self.tr = np.array((np.max(x), np.max(y)))

            self.x_init = np.array(self.proj(*(Utils.RAD_TO_DEG * self.x_init)))

        self.delta_x = (self.tr[0] - self.bl[0]) / (self.nx - 1)
        self.delta_y = (self.tr[1] - self.bl[1]) / (self.ny - 1)

        self.speed = mp.model.v_a
        self.wind = None
        self.phi = None

        self.ts = None
        self.flatten = False
        self.flatten_factor = Utils.EARTH_RADIUS

        self.method = method

        self.matlabLS_path = None

        self.t = 0
        self.kernel = kernel
        if self.kernel not in ['base', 'matlab']:
            print(f'Unknown kernel : {self.kernel}')
            exit(1)

    def setup_matlab(self):
        path = os.environ.get('DABRYPATH')
        if path is None:
            raise Exception('Unable to locate matlab script. Please set DABRYPATH variable.')
        self.matlabLS_path = os.path.join(path, 'rft')

    def compute(self):
        if self.kernel == 'base':
            # Setting up wind array
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nt):
                        self.wind[i, j, k, :] = self.mp.model.wind.value(
                            self.bl + np.array([self.delta_x * i, self.delta_y * j]))
            # Setting RF function to distance function at t=0
            for i in range(self.nx):
                for j in range(self.ny):
                    point = self.bl + np.array([self.delta_x * i, self.delta_y * j])
                    self.phi[i, j, 0] = np.linalg.norm(self.mp.x_init - point) - 5e-3
            # Integrate
            for k in range(self.nt - 1):
                self.update_phi()

        elif self.kernel == 'matlab':

            # Give all arguments to matlab in a JSON file
            epsx = (self.tr[0] - self.bl[0]) * 0.005
            epsy = (self.tr[1] - self.bl[1]) * 0.005
            args = {
                "nx": self.nx,
                "ny": self.ny,
                "x_min": self.bl[0] + epsx,
                "x_max": self.tr[0] - epsx,
                "y_min": self.bl[1] + epsy,
                "y_max": self.tr[1] - epsy,
                "x_init": self.x_init[0],
                "y_init": self.x_init[1],
                "airspeed": self.mp.model.v_a,
                "t_start": self.mp.model.wind.t_start,
                "t_end": self.mp.model.wind.t_start + self.max_time,
                "nt_rft": self.nt
            }
            with open(os.path.join(self.matlabLS_path, 'input', 'args.json'), 'w') as f:
                json.dump(args, f)

            # Give also the wind to the matlab script
            uv = np.zeros((self.nt, self.nx, self.ny, 2))

            grid = np.zeros((self.nx, self.ny, 2))
            for i in range(self.nx):
                for j in range(self.ny):
                    point = np.array([self.bl[0] + i * self.delta_x, self.bl[1] + j * self.delta_y])
                    if self.mp.coords == Utils.COORD_GCS:
                        point = Utils.DEG_TO_RAD * np.array(self.proj(*point, inverse=True))
                    grid[i, j, :] = point

            ts = np.linspace(self.mp.model.wind.t_start, self.mp.model.wind.t_start + self.max_time, self.nt)

            t0 = self.mp.model.wind.t_start
            if self.mp.model.wind.t_end is None:
                delta_t = 0.
            else:
                delta_t = (self.mp.model.wind.t_end - self.mp.model.wind.t_start) / (self.nt - 1)
            for k in range(self.nt):
                t = t0 + k * delta_t
                for i in range(self.nx):
                    for j in range(self.ny):
                        point = np.zeros(2)
                        point[:] = grid[i, j]
                        value = self.mp.model.wind.value(t, point)
                        uv[k, i, j, :] = value

            if self.mp.coords == Utils.COORD_GCS:
                dwind = DiscreteWind(wdata={'data': uv, 'grid': grid, 'ts': ts, 'coords': self.mp.coords})
                m = 0.5 * (self.mp.x_init + self.mp.x_target)
                dwind.flatten(proj='ortho', lon_0=m[0], lat_0=m[1])
                # dwind.flatten(proj='omerc',
                #               lon_1=self.mp.x_init[0],
                #               lat_1=self.mp.x_init[1],
                #               lon_2=self.mp.x_target[0],
                #               lat_2=self.mp.x_target[1])
                uv = dwind.uv
            uv = uv.transpose((3, 1, 2, 0))
            d = {'data': uv}
            scipy.io.savemat(os.path.join(self.matlabLS_path, 'input', 'wind.mat'), d)
            """
            np.savetxt(os.path.join(self.matlabLS_path, 'input', 'u.txt'), u, delimiter=",")
            np.savetxt(os.path.join(self.matlabLS_path, 'input', 'v.txt'), v, delimiter=",")
            """

            # Run matlab script
            os.system(f'matlab -sd {self.matlabLS_path} -batch "main; exit"')

            # Load matlab output in corresponding fields
            self.ts = np.genfromtxt(os.path.join(self.matlabLS_path, 'output', 'timesteps.txt'), delimiter=',')
            self.nt = self.ts.shape[0]
            self.nx, self.ny = np.genfromtxt(os.path.join(self.matlabLS_path, 'output', 'rff_0.txt'),
                                             delimiter=',').shape
            self.phi = np.zeros((self.nx, self.ny, self.nt))
            for k in range(self.nt):
                self.phi[:, :, k] = np.genfromtxt(os.path.join(self.matlabLS_path, 'output', f'rff_{k}.txt'),
                                                  delimiter=',')

        elif self.kernel == 'cache':
            pass

    def load_cache(self, filepath):
        try:
            print('[RFT] Using cached values')
            with h5py.File(filepath, 'r') as f:
                nt, nx, ny = f['data'].shape
                self.phi = np.zeros((nx, ny, nt))
                self.mp.coords = f.attrs['coords']
                self.phi[:, :, :] = np.array(f['data']).transpose((1, 2, 0))

                self.max_time = np.array(f['ts']).max()

                grid = np.zeros(f['grid'].shape)
                grid[:] = f['grid']

                factor = Utils.DEG_TO_RAD if self.mp.coords == Utils.COORD_GCS else 1.
                self.bl = factor * np.array((grid[:, :, 0].min(), grid[:, :, 1].min()))
                self.tr = factor * np.array((grid[:, :, 0].max(), grid[:, :, 1].max()))

                self.kernel = 'cache'
        except FileNotFoundError:
            print(f'[RFT] Could not load cache, resuming with usual computation ({filepath} : Not found)',
                  file=sys.stderr)

    def update_phi(self):
        if self.method == 'lolla':
            # Copy current phi
            phi_cur = np.zeros(self.phi.shape[:-1])
            phi_cur[:] = self.phi[:, :, self.t]

            # Step 1 : build phi star
            phi_star = np.zeros(phi_cur.shape)
            phi_star[:] = phi_cur - self.delta_t / 2 * self.speed * norm_grad(phi_cur, self.delta_x, self.delta_y,
                                                                              method='central')

            # Step 2 : build phi star star
            phi_sstar = np.zeros(phi_cur.shape)
            phi_sstar[:] = phi_star - self.delta_t * np.einsum('ijk,ijk->ij', self.wind[:, :, self.t, :],
                                                               grad(phi_star, self.delta_x, self.delta_y,
                                                                    method='central'))

            # Step 3 : build next phi
            self.phi[:, :, self.t + 1] = phi_sstar - self.delta_t / 2. * self.speed * norm_grad(phi_sstar,
                                                                                                self.delta_x,
                                                                                                self.delta_y,
                                                                                                method='central')

        elif self.method == 'sethian':
            phi_cur = np.zeros(self.phi.shape[:-1])
            phi_cur[:] = self.phi[:, :, self.t]

            self.phi[:, :, self.t + 1] = phi_cur - self.delta_t * (
                    self.speed * norm_grad(phi_cur, self.delta_x, self.delta_y, method='central') + wind_dot_phi(
                phi_cur, self.wind[:, :, self.t, :], self.delta_x))

        else:
            print(f'Unknown update method for phi : {self.method}')

        self.t += 1

    def dump_rff(self, filepath, lon0=None, lat0=None):
        with h5py.File(os.path.join(filepath, 'rff.h5'), 'w') as f:
            nx, ny, nt = self.phi.shape
            f.attrs['coords'] = self.mp.coords
            dset = f.create_dataset('data', (nt, nx, ny), dtype='f8')
            dset[:, :, :] = self.phi.transpose((2, 0, 1))

            dset = f.create_dataset('ts', (nt,), dtype='f8')
            dset[:] = np.linspace(self.mp.model.wind.t_start, self.mp.model.wind.t_start + self.max_time, nt)

            dset = f.create_dataset('grid', (nx, ny, 2), dtype='f8')
            X, Y = np.meshgrid(self.bl[0] + self.delta_x * np.arange(self.nx),
                               self.bl[1] + self.delta_y * np.arange(self.ny), indexing='ij')
            if self.mp.coords == Utils.COORD_GCS:
                X, Y = self.proj(X, Y, inverse=True)
                X[:] = Utils.DEG_TO_RAD * X
                Y[:] = Utils.DEG_TO_RAD * Y
            dset[:, :, 0] = X
            dset[:, :, 1] = Y

    def build_front(self, point):
        k = self.get_index(point) - 1
        # TODO : make that more general
        # For the moment, store it in first front
        lam = self.value(point, k + 1) / (self.value(point, k + 1) - self.value(point, k))
        self.phi[:, :, 0] = lam * self.phi[:, :, k] + (1 - lam) * self.phi[:, :, k + 1]

    def project(self, x):
        if self.mp.coords == Utils.COORD_CARTESIAN:
            return x
        else:
            return np.array(self.proj(*(Utils.RAD_TO_DEG * np.array(x))))

    def _value(self, point, timeindex):
        """
        Compute phi value at given point and timestamp using linear interpolation
        :param point: Point at which to compute the front function
        :param timeindex: The time index
        :return: Value of phi
        """
        nx, ny, nt = self.phi.shape
        # Find the tile to which requested point belongs
        # A tile is a region between 4 known values of wind
        xx = (point[0] - self.bl[0]) / (self.tr[0] - self.bl[0])
        yy = (point[1] - self.bl[1]) / (self.tr[1] - self.bl[1])
        delta_x = (self.tr[0] - self.bl[0]) / (nx - 1)
        delta_y = (self.tr[1] - self.bl[1]) / (ny - 1)
        i = int((nx - 1) * xx)
        j = int((ny - 1) * yy)
        # Tile bottom left corner x-coordinate
        xij = delta_x * i + self.bl[0]
        # Point relative position in tile
        a = (point[0] - xij) / delta_x
        # Tile bottom left corner y-coordinate
        yij = delta_y * j + self.bl[1]
        b = (point[1] - yij) / delta_y

        return (1 - b) * ((1 - a) * self.phi[i, j, timeindex] + a * self.phi[i + 1, j, timeindex]) + b * (
                (1 - a) * self.phi[i, j + 1, timeindex] + a * self.phi[i + 1, j + 1, timeindex])

    def value(self, t, x):
        x = self.project(x)
        k, alpha = self._index(t)
        return (1 - alpha) * self._value(x, k) + alpha * self._value(x, k + 1)

    def control(self, x, backward=False):
        factor = 1. if backward else -1.
        xp = self.project(x)
        s = factor * self.get_normal(xp)[0]
        if self.mp.coords == Utils.COORD_CARTESIAN:
            return atan2(s[1], s[0])
        else:
            # Control vector is in projected space
            # It has to be rotated to fit to spherical space
            tmp = Utils.d_proj_ortho_inv(xp[0], xp[1], self.lon_0, self.lat_0) @ s
            s_new = s  # / np.linalg.norm(tmp)
            # s_new = tmp / np.linalg.norm(tmp)
            return np.pi / 2. - atan2(s_new[1], s_new[0])

    def backward_traj(self, point, new_target, ceil, T, model, N_disc=100):
        """
        Build backward trajectory starting from given endpoint
        :param point: Endpoint
        :param model: Zermelo model object
        :return: Trajectory integrated backwards using normal to fronts
        """
        x0 = np.zeros(2)
        x0[:] = point

        def f(x, t):
            u = self.control(x)
            return - model.dyn.value(x, u, 0.)

        timestamps = np.linspace(0., T, N_disc)
        dt = T / N_disc
        points = np.zeros((N_disc, 2))
        points[0, :] = point
        k = 0
        for k in range(1, N_disc):
            points[k, :] = f(points[k - 1, :], 0.) * dt + points[k - 1, :]
            if Utils.distance(points[k, :], new_target, coords=self.mp.coords) < ceil:
                break
        # points = np.array(odeint(f, x0, timestamps))
        controls = np.array(list(map(self.control, points)))
        return Trajectory(timestamps, points[::-1], controls, k, optimal=True, coords=self.mp.coords)

    def get_index(self, point):
        """
        Get first time index for which given point is in a front
        :param point: Point to check
        :return: First index such that phi is negative at that index and point
        """
        nx, ny, nt = self.phi.shape
        # Find the tile to which requested point belongs
        # A tile is a region between 4 known values of wind
        xx = (point[0] - self.bl[0]) / (self.tr[0] - self.bl[0])
        yy = (point[1] - self.bl[1]) / (self.tr[1] - self.bl[1])
        if xx < 0. or xx > 1. or yy < 0. or yy > 1.:
            print(f'Point not within bounds {tuple(point)}')
        delta_x = (self.tr[0] - self.bl[0]) / (nx - 1)
        delta_y = (self.tr[1] - self.bl[1]) / (ny - 1)
        i = int((nx - 1) * xx)
        j = int((ny - 1) * yy)
        if i == nx - 1:
            i -= 1
        if j == ny - 1:
            j -= 1
        # Tile bottom left corner x-coordinate
        xij = delta_x * i + self.bl[0]
        # Point relative position in tile
        a = (point[0] - xij) / delta_x
        # Tile bottom left corner y-coordinate
        yij = delta_y * j + self.bl[1]
        b = (point[1] - yij) / delta_y
        phi_itp = 0.
        try:
            for k in range(1, nt):
                phi_itp = (1 - b) * ((1 - a) * self.phi[i, j, k] + a * self.phi[i + 1, j, k]) + \
                          b * ((1 - a) * self.phi[i, j + 1, k] + a * self.phi[i + 1, j + 1, k])
                if phi_itp < 0.:
                    return k
            if phi_itp > 0.:
                print(f'[Init shooting] Error: Point is not within any front. ({point[0]}, {point[1]})',
                      file=sys.stderr)
                return -1
        except IndexError as e:
            raise e

    def get_normal(self, point, index=None, compute=False):
        if compute:
            self.compute()
        nx, ny, nt = self.phi.shape
        # Find the tile to which requested point belongs
        # A tile is a region between 4 known values of wind
        xx = (point[0] - self.bl[0]) / (self.tr[0] - self.bl[0])
        yy = (point[1] - self.bl[1]) / (self.tr[1] - self.bl[1])
        delta_x = (self.tr[0] - self.bl[0]) / (nx - 1)
        delta_y = (self.tr[1] - self.bl[1]) / (ny - 1)
        i = int((nx - 1) * xx)
        j = int((ny - 1) * yy)
        # Tile bottom left corner x-coordinate
        xij = delta_x * i + self.bl[0]
        # Point relative position in tile
        a = (point[0] - xij) / delta_x
        # Tile bottom left corner y-coordinate
        yij = delta_y * j + self.bl[1]
        b = (point[1] - yij) / delta_y

        if index is not None:
            k = index
        else:
            k = self.get_index(point) - 1

        lam = self._value(point, k + 1) / (self._value(point, k + 1) - self._value(point, k))

        # Linear interpolation gradient at point gives the normal
        d_phi__d_x = 1. / delta_x * ((1 - b) * (self.phi[i + 1, j, k] - self.phi[i, j, k]) +
                                     b * (self.phi[i + 1, j + 1, k] - self.phi[i, j + 1, k]))
        d_phi__d_y = 1. / delta_y * ((1 - a) * (self.phi[i, j + 1, k] - self.phi[i, j, k]) +
                                     a * (self.phi[i + 1, j + 1, k] - self.phi[i + 1, j, k]))
        d_phi_k = np.array((d_phi__d_x, d_phi__d_y))
        d_phi__d_x = 1. / delta_x * ((1 - b) * (self.phi[i + 1, j, k + 1] - self.phi[i, j, k + 1]) +
                                     b * (self.phi[i + 1, j + 1, k + 1] - self.phi[i, j + 1, k + 1]))
        d_phi__d_y = 1. / delta_y * ((1 - a) * (self.phi[i, j + 1, k + 1] - self.phi[i, j, k + 1]) +
                                     a * (self.phi[i + 1, j + 1, k + 1] - self.phi[i + 1, j, k + 1]))
        d_phi_kp1 = np.array((d_phi__d_x, d_phi__d_y))
        d_phi = lam * d_phi_k / np.linalg.norm(d_phi_k) + (1 - lam) * d_phi_kp1 / np.linalg.norm(d_phi_kp1)
        return d_phi / np.linalg.norm(d_phi), self.max_time * (k + 1) / self.nt, k + 1

    def has_reached(self):
        return self.value(self.ts[-1], self.mp.x_target) < 0.

    def get_time(self, x):
        """
        Get time for which phi(t, x) = 0 by dichotomia
        :param x: The point at which to fetch the time value
        :return: The time value
        """
        eps = 1e-4
        tl = self.ts[0]
        tu = self.ts[-1]
        while tu - tl > eps:
            t_new = 0.5 * (tl + tu)
            if self.value(t_new, x) > 0.:
                tl = t_new
            else:
                tu = t_new
        return tl

    def _index(self, t):
        if t > self.ts[-1]:
            return self.nt - 2, 1.
        if t < self.ts[0]:
            return 0, 0.
        else:
            tt = (t - self.ts[0]) / (self.ts[-1] - self.ts[0])
            if int(tt * (self.nt - 1)) == self.nt - 1:
                return self.nt - 2, 1.
            return int(tt * (self.nt - 1)), tt * (self.nt - 1) - int(tt * (self.nt - 1))


if __name__ == '__main__':
    nx = 300
    ny = 300
    nts = 30
    delta_x = 1 / (nx - 1)
    delta_y = 1 / (ny - 1)
    delta_t = 0.1

    gradient = np.array([1.0, 2.0])
    field = np.zeros((nx, ny))
    norm_grad_field = np.zeros((nx, ny))
    d_field__d_x = np.zeros((nx, ny))
    d_field__d_y = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            x, y = delta_x * i, delta_y * j
            field[i, j] = sin(2 * np.pi * (x + 0.5 * y))
            d_field__d_x[i, j] = 2 * np.pi * cos(2 * np.pi * (x + 0.5 * y))
            d_field__d_y[i, j] = np.pi * cos(2 * np.pi * (x + 0.5 * y))
            norm_grad_field[i, j] = np.sqrt(
                2 * np.pi * cos(2 * np.pi * (x + 0.5 * y)) ** 2 + np.pi * cos(2 * np.pi * (x + 0.5 * y)) ** 2)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    pos = ax[0, 0].imshow(field)
    ax[0, 0].set_title('$f$')
    fig.colorbar(pos, ax=ax[0, 0])
    pos = ax[0, 1].imshow(d_field__d_x)
    ax[0, 1].set_title('$\partial_x f$')
    fig.colorbar(pos, ax=ax[0, 1])
    pos = ax[1, 0].imshow((d_field__d_x - central_diff(field, 0, delta_x))[1:-1, 1:-1])
    ax[1, 0].set_title('Central diff delta')
    fig.colorbar(pos, ax=ax[1, 0])
    pos = ax[1, 1].imshow((d_field__d_x - upwind_diff(field, 0, delta_x))[1:-1, 1:-1])
    ax[1, 1].set_title('Upwind delta')
    # pos = ax[1].imshow((d_field__d_x - upwind_diff(field, 0, delta_x))[1:-1, 1:-1])
    fig.colorbar(pos, ax=ax[1, 1])

    print(f'Central max diff x    : {np.max(np.abs((d_field__d_x - central_diff(field, 0, delta_x))[1:-1, 1:-1]))}')
    print(
        f'Central max relative diff x: {np.max(np.abs((1 - central_diff(field, 0, delta_x) / d_field__d_x)[1:-1, 1:-1]))}')
    print(f'Upwind max diff x    : {np.max(np.abs((d_field__d_x - upwind_diff(field, 0, delta_x))[1:-1, 1:-1]))}')
    print(
        f'Upwind max relative diff x: {np.max(np.abs((1 - upwind_diff(field, 0, delta_x) / d_field__d_x)[1:-1, 1:-1]))}')

    print(f'Central max diff y    : {np.max(np.abs((d_field__d_y - central_diff(field, 1, delta_y))[1:-1, 1:-1]))}')
    print(
        f'Central max relative diff y: {np.max(np.abs((1 - central_diff(field, 1, delta_y) / d_field__d_y)[1:-1, 1:-1]))}')
    print(f'Upwind max diff y    : {np.max(np.abs((d_field__d_y - upwind_diff(field, 1, delta_y))[1:-1, 1:-1]))}')
    print(
        f'Upwind max relative diff y: {np.max(np.abs((1 - upwind_diff(field, 1, delta_y) / d_field__d_y)[1:-1, 1:-1]))}')

    plt.show()

    # origin + np.array([delta_x * i, delta_y * j])
    # field = np.array()
