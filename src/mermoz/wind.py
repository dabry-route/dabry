import sys
import warnings

import numpy as np
import scipy.interpolate as itp

import h5py
from mpl_toolkits.basemap import Basemap
from math import exp, log
from pyproj import Proj

from mermoz.misc import *


class Wind:

    def __init__(self, value_func=None, d_value_func=None, descr='', is_analytical=False, t_start=0.,
                 t_end=None, nt=1, **kwargs):
        """
        Builds a windfield which is a smooth vector field of space
        value_func Function taking a point in space (ndarray) and returning the
        wind value at the given point.
        d_value_func Function taking a point in space (ndarray) and returning the
        jacobian of the windfield at the given point.
        """
        self.value = value_func
        self.d_value = d_value_func
        self.descr = descr
        # 0 if wind is not dumpable,
        # 1 if dumpable but needs sampling (analytical winds)
        # 2 if directly dumpable
        self.is_dumpable = 1

        # Used when wind is flattened
        self.lon_0 = None
        self.lat_0 = None

        # When grid is not regularly spaced
        self.unstructured = False

        # When wind is time-varying, bounds for the time window
        # A none upper bound means no time variation
        self.t_start = t_start
        self.t_end = t_end
        self.nt = nt

        # To remember if an analytical wind was discretized for
        # display purposes.
        self.is_analytical = is_analytical

        # True if dualization operation flips wind (mulitplication by -1)
        # False if dualization operation leaves wind unchanged
        self.dualizable = True

        for k, v in kwargs.items():
            self.__dict__[k] = v

    def _index(self, t):
        """
        Get nearest lowest index for time discrete grid
        :param t: The required time
        :return: Nearest lowest index for time, coefficient for the linear interpolation
        """
        if self.t_end is None:
            return 0, 0.
        if self.nt == 1:
            return 0, 0.
        # if t < self.t_start or t > self.t_end:
        #     print(f'Time {t} out of ({self.t_start}, {self.t_end})', file=sys.stderr)
        #     exit(1)
        # Bounds may not be in the right order if wind is dualized
        t_min, t_max = min(self.t_start, self.t_end), max(self.t_start, self.t_end)
        if t <= t_min:
            # print(f'Time {t} lower than lower time bound {t_min}', file=sys.stderr)
            # warnings.warn(UserWarning('Probably considering wind as steady'))
            return 0, 0.
        if t > t_max:
            # Freeze wind to last frame
            return self.nt - 2, 1.
        tau = (t - t_min) / (t_max - t_min)
        i, alpha = int(tau * (self.nt - 1)), tau * (self.nt - 1) - int(tau * (self.nt - 1))
        if i == self.nt - 1:
            i = self.nt - 2
            alpha = 1.
        return i, alpha

    def dualize(self):
        a = -1.
        if self.t_end is not None:
            mem = self.t_start
            self.t_start = self.t_end
            self.t_end = mem
        if not self.dualizable:
            a = 1.
        return a * self

    def __add__(self, other):
        """
        Add windfields

        :param other: Another windfield
        :return: The sum of the two windfields
        """
        if not isinstance(other, Wind):
            raise TypeError(f"Unsupported type for addition : {type(other)}")
        value_func = lambda t, x: self.value(t, x) + other.value(t, x)
        d_value_func = lambda t, x: self.d_value(t, x) + other.d_value(t, x)
        descr = self.descr + ', ' + other.descr
        is_analytical = self.is_analytical and other.is_analytical
        # d = self.__dict__
        # for k in ['value_func', 'd_value_func', 'descr', 'is_analytical']:
        #     try:
        #         del d[k]
        #     except KeyError:
        #         pass
        return Wind(value_func=value_func,
                    d_value_func=d_value_func,
                    descr=descr,
                    is_analytical=is_analytical)
        # **d)

    def __mul__(self, other):
        """
        Handles the scaling of a windfield by a real number

        :param other: A real number (float)
        :return: The scaled windfield
        """
        if isinstance(other, int):
            other = float(other)
        if not isinstance(other, float):
            raise TypeError(f"Unsupported type for multiplication : {type(other)}")
        value_func = lambda t, x: other * self.value(t, x)
        d_value_func = lambda t, x: other * self.d_value(t, x)
        descr = self.descr
        is_analytical = self.is_analytical
        # d = self.__dict__
        # for k in ['value_func', 'd_value_func', 'descr', 'is_analytical']:
        #     try:
        #         del d[k]
        #     except KeyError:
        #         pass
        return Wind(value_func=value_func,
                    d_value_func=d_value_func,
                    descr=descr,
                    is_analytical=is_analytical,
                    t_start=self.t_start,
                    t_end=self.t_end)
        # **d)

    def __rmul__(self, other):
        return self.__mul__(other)

    def e_minus(self, x):
        """
        Returns an eigenvector of the vector space associated to the
        negative eigenvalue of the jacobian of the windfield

        :param x: The point in space where the eigenvector is computed
        :return: The eigenvector
        """
        d_wind = self.d_value(x)
        dw_x = d_wind[0, 0]
        dw_y = d_wind[0, 1]
        lambda_0 = np.sqrt(dw_x ** 2 + dw_y ** 2)
        return np.array([- dw_y, lambda_0 + dw_x])

    def grad_norm(self, x):
        """
        Returns the gradient of the wind norm function at given point

        :param x: The point at which to compute the gradient
        :return: The gradient
        """
        wind = self.value(x)
        return np.linalg.norm(self.d_value(x).transpose().dot(wind / np.linalg.norm(wind)))

    def __str__(self):
        return self.descr


class DiscreteWind(Wind):
    """
    Handles wind loading from H5 format and derivative computation
    """

    def __init__(self, interp='pwc', force_analytical=False, wdata=None, diff=False):
        super().__init__(value_func=self.value, d_value_func=self.d_value)

        self.is_dumpable = 2

        if wdata is None:
            self.nt = None
            self.nx = None
            self.ny = None

            self.uv = None
            self.grid = None
            self.coords = None
            self.ts = None

            self.x_min = None
            self.x_max = None
            self.y_min = None
            self.y_max = None

            self.units_grid = None

            self.d_u__d_x = None
            self.d_u__d_y = None
            self.d_v__d_x = None
            self.d_v__d_y = None
        else:
            if len(wdata['data'].shape) == 2:
                self.nt = None
                self.nx, self.ny, _ = wdata['data'].shape
            else:
                self.nt, self.nx, self.ny, _ = wdata['data'].shape

            self.uv = np.zeros(wdata['data'].shape)
            self.uv[:] = wdata['data']
            self.grid = np.zeros(wdata['grid'].shape)
            self.grid[:] = wdata['grid']
            self.ts = np.zeros(wdata['ts'].shape)
            self.ts[:] = wdata['ts']
            self.coords = wdata['coords']

            self.x_min = np.min(self.grid[:, :, 0])
            self.x_max = np.max(self.grid[:, :, 0])
            self.y_min = np.min(self.grid[:, :, 1])
            self.y_max = np.max(self.grid[:, :, 1])

            self.units_grid = U_RAD if self.coords == COORD_GCS else COORD_CARTESIAN

            self.d_u__d_x = None
            self.d_u__d_y = None
            self.d_v__d_x = None
            self.d_v__d_y = None

        # Either 'pwc' for piecewise constant wind or 'linear' for linear interpolation
        self.interp = interp

        self.clipping_tol = 1e-2

        self.is_analytical = force_analytical

        self.bm = None

        if wdata is not None and diff:
            self.compute_derivatives()

    def load(self, filepath, nodiff=False, resample=1, duration_limit=None):
        """
        Loads wind data from H5 wind data
        :param filepath: The H5 file contaning wind data
        :param nodiff: Do not compute wind derivatives
        :param resample: If unstructured, the resample rate over which to evaluate wind
        """

        # Fill the wind data array
        print(f'{"Loading wind values...":<30}', end='')
        with h5py.File(filepath, 'r') as wind_data:
            self.coords = wind_data.attrs['coords']
            self.units_grid = wind_data.attrs['units_grid']
            try:
                self.lon_0 = wind_data.attrs['lon_0']
                self.lat_0 = wind_data.attrs['lat_0']
            except KeyError:
                pass
            try:
                # In a try/except for backward compatibility
                self.unstructured = wind_data.attrs['unstructured']
            except KeyError:
                pass

            # Checking consistency before loading
            ensure_coords(self.coords)
            ensure_units(self.units_grid)
            ensure_compatible(self.coords, self.units_grid)

            # Define computation bounds as smallest xy-coordinate outer bounding box
            self.x_min = wind_data['grid'][:, :, 0].min()
            self.x_max = wind_data['grid'][:, :, 0].max()
            self.y_min = wind_data['grid'][:, :, 1].min()
            self.y_max = wind_data['grid'][:, :, 1].max()

            # Loading
            if not self.unstructured:
                self.nt, self.nx, self.ny, _ = wind_data['data'].shape
                # if duration_limit is not None:
                #     new_nt = 0
                #     for t in wind_data['ts']:
                #         if t - wind_data['ts'][0] > duration_limit:
                #             break
                #         new_nt += 1
                # self.nt = new_nt
                self.uv = np.zeros((self.nt, self.nx, self.ny, 2))
                self.uv[:, :, :, :] = wind_data['data'][:self.nt, :, :, :]
                self.grid = np.zeros((self.nx, self.ny, 2))
                self.grid[:] = wind_data['grid']
                self.ts = np.zeros((self.nt,))
                self.ts[:] = wind_data['ts'][:self.nt]
                self.t_start = self.ts[0]
                self.t_end = None if self.nt == 1 else self.ts[-1]
                # Detecting millisecond-formated timestamps
                if np.any(self.ts > 1e11):
                    self.ts[:] = self.ts / 1000.
                    self.t_start /= 1000.
                    if self.t_end is not None:
                        self.t_end /= 1000.

            else:
                nx_wind = wind_data['data'].shape[1]
                ny_wind = wind_data['data'].shape[2]
                self.nt = wind_data['data'].shape[0]
                self.nx = resample * nx_wind
                self.ny = resample * ny_wind
                self.grid = np.zeros((self.nx, self.ny, 2))
                self.grid[:] = np.array(np.meshgrid(np.linspace(self.x_min, self.x_max, self.nx),
                                                    np.linspace(self.y_min, self.y_max, self.ny))).transpose((2, 1, 0))
                grid_wind = np.zeros(wind_data['grid'].shape)
                grid_wind[:] = np.array(wind_data['grid'])
                u_wind = np.zeros((nx_wind, ny_wind))
                v_wind = np.zeros((nx_wind, ny_wind))
                u_wind[:] = np.array(wind_data['data'][0, :, :, 0])
                v_wind[:] = np.array(wind_data['data'][0, :, :, 1])
                u_itp = np.zeros((self.nx, self.ny))
                u_itp[:] = itp.griddata(grid_wind.reshape((nx_wind * ny_wind, 2)),
                                        u_wind.reshape((nx_wind * ny_wind,)),
                                        self.grid.reshape((self.nx * self.ny, 2)), method='linear',
                                        fill_value=0.).reshape((self.nx, self.ny))
                v_itp = np.zeros((self.nx, self.ny))
                v_itp[:] = itp.griddata(grid_wind.reshape((nx_wind * ny_wind, 2)),
                                        v_wind.reshape((nx_wind * ny_wind,)),
                                        self.grid.reshape((self.nx * self.ny, 2)), method='linear',
                                        fill_value=0.).reshape((self.nx, self.ny))
                self.uv = np.zeros((self.nt, self.nx, self.ny, 2))
                for kt in range(self.nt):
                    self.uv[kt, :] = np.stack((u_itp, v_itp), axis=2)
                self.ts = np.zeros((self.nt,))
                self.ts[:] = wind_data['ts']

        # Post processing
        if self.units_grid == U_DEG:
            self.grid[:] = DEG_TO_RAD * self.grid
            self.x_min, self.x_max, self.y_min, self.y_max = \
                tuple(DEG_TO_RAD * np.array((self.x_min, self.x_max, self.y_min, self.y_max)))

            self.units_grid = U_RAD

        # self.x_min = np.max(self.grid[0, :, 0])
        # self.x_max = np.min(self.grid[-1, :, 0])
        # self.y_min = np.max(self.grid[:, 0, 1])
        # self.y_max = np.min(self.grid[:, -1, 1])

        if not nodiff:
            self.compute_derivatives()

        print('Done')

    def load_from_wind(self, wind: Wind, nx, ny, bl, tr, coords, nodiff=False, nt=1, fd=False):
        self.coords = coords
        self.units_grid = 'meters' if coords == COORD_CARTESIAN else 'degrees'
        self.unstructured = wind.unstructured

        # Checking consistency before loading
        ensure_units(self.units_grid)
        ensure_coords(self.coords)
        ensure_compatible(self.coords, self.units_grid)

        self.x_min = bl[0]
        self.x_max = tr[0]
        self.y_min = bl[1]
        self.y_max = tr[1]

        self.t_start = wind.t_start
        self.t_end = wind.t_end

        self.nt, self.nx, self.ny = nt, nx, ny
        self.uv = np.zeros((self.nt, self.nx, self.ny, 2))
        self.grid = np.zeros((self.nx, self.ny, 2))
        delta_x = (self.x_max - self.x_min) / (self.nx - 1)
        delta_y = (self.y_max - self.y_min) / (self.ny - 1)
        self.ts = np.zeros((self.nt,))
        for k in range(nt):
            if wind.t_end is not None:
                self.ts[k] = wind.t_start + (k / (self.nt - 1)) * (wind.t_end - wind.t_start)
            for i in range(self.nx):
                for j in range(self.ny):
                    point = np.array([self.x_min + i * delta_x, self.y_min + j * delta_y])
                    if wind.t_end is not None:
                        self.uv[k, i, j, :] = wind.value(self.ts[k], point)
                    else:
                        self.uv[k, i, j, :] = wind.value(wind.t_start, point)
                    self.grid[i, j, :] = point

        # Post processing
        # Loading derivatives
        if not nodiff:
            if not fd:
                self.d_u__d_x = np.zeros((self.nt, self.nx - 2, self.ny - 2))
                self.d_u__d_y = np.zeros((self.nt, self.nx - 2, self.ny - 2))
                self.d_v__d_x = np.zeros((self.nt, self.nx - 2, self.ny - 2))
                self.d_v__d_y = np.zeros((self.nt, self.nx - 2, self.ny - 2))
                for k in range(self.nt):
                    for i in range(1, self.nx - 1):
                        for j in range(1, self.ny - 1):
                            point = np.array([self.x_min + i * delta_x, self.y_min + j * delta_y])
                            if nt > 1:
                                diff = wind.d_value(self.ts[k], point)
                            else:
                                diff = wind.d_value(wind.t_start, point)
                            self.d_u__d_x[0, i - 1, j - 1] = diff[0, 0]
                            self.d_u__d_y[0, i - 1, j - 1] = diff[0, 1]
                            self.d_v__d_x[0, i - 1, j - 1] = diff[1, 0]
                            self.d_v__d_y[0, i - 1, j - 1] = diff[1, 1]
            else:
                self.compute_derivatives()

    def value(self, t, x, units=U_METERS):
        """
        :param x: Point at which to give wind interpolation
        :param units: Units in which x is given
        :return: The interpolated wind value at x
        """
        # ensure_units(units)
        # ensure_compatible(self.coords, units)

        nx = self.nx
        ny = self.ny

        xx = (x[0] - self.x_min) / (self.x_max - self.x_min)
        yy = (x[1] - self.y_min) / (self.y_max - self.y_min)

        it, alpha = self._index(t)

        eps = self.clipping_tol
        if xx < 0. - eps or xx >= 1. or yy < 0. - eps or yy >= 1.:
            # print(f"Real windfield undefined at ({x[0]:.3f}, {x[1]:.3f})")
            return np.array([0., 0.])

        if self.interp == 'pwc':
            try:
                if self.nt == 1:
                    return self.uv[0, int((nx - 1) * xx + 0.5), int((ny - 1) * yy + 0.5)]
                else:
                    return (1 - alpha) * self.uv[it, int((nx - 1) * xx + 0.5), int((ny - 1) * yy + 0.5)] + \
                           alpha * self.uv[it + 1, int((nx - 1) * xx + 0.5), int((ny - 1) * yy + 0.5)]
            except ValueError:
                return np.zeros(2)

        elif self.interp == 'linear':
            # Find the tile to which requested point belongs
            # A tile is a region between 4 known values of wind
            delta_x = (self.x_max - self.x_min) / (nx - 1)
            delta_y = (self.y_max - self.y_min) / (ny - 1)
            i = int((nx - 1) * xx)
            j = int((ny - 1) * yy)
            # Tile bottom left corner x-coordinate
            xij = delta_x * i + self.x_min
            # Point relative position in tile
            a = (x[0] - xij) / delta_x
            # Tile bottom left corner y-coordinate
            yij = delta_y * j + self.y_min
            b = (x[1] - yij) / delta_y
            if self.nt == 1:
                return (1 - b) * ((1 - a) * self.uv[0, i, j] + a * self.uv[0, i + 1, j]) + b * (
                        (1 - a) * self.uv[0, i, j + 1] + a * self.uv[0, i + 1, j + 1])
            else:
                return (1 - alpha) * ((1 - b) * ((1 - a) * self.uv[it, i, j] + a * self.uv[it, i + 1, j]) +
                                      b * ((1 - a) * self.uv[it, i, j + 1] + a * self.uv[it, i + 1, j + 1])) + \
                       alpha * ((1 - b) * ((1 - a) * self.uv[it + 1, i, j] + a * self.uv[it + 1, i + 1, j]) +
                                b * ((1 - a) * self.uv[it + 1, i, j + 1] + a * self.uv[it + 1, i + 1, j + 1]))

    def d_value(self, t, x, units=U_METERS):
        """
        :param x: Point at which to give wind interpolation
        :param units: Units in which x is given
        :return: The interpolated wind value at x
        """
        # ensure_units(units)
        # ensure_compatible(self.coords, units)

        nx = self.nx
        ny = self.ny

        it, alpha = self._index(t)

        factor = 1.  # (DEG_TO_RAD if self.coords == COORD_GCS and units == U_DEG else 1.)

        xx = (x[0] - self.x_min) / (self.x_max - self.x_min)
        yy = (x[1] - self.y_min) / (self.y_max - self.y_min)

        eps = self.clipping_tol
        if xx < 1 / (2 * (nx - 1)) - eps or \
                xx >= 1. - 1 / (2 * (nx - 1)) + eps or \
                yy < 1 / (2 * (ny - 1)) - eps or yy >= 1. - 1 / (2 * (ny - 1)) + eps:
            # print(f"Real windfield jacobian undefined at ({x[0]:.3f}, {x[1]:.3f})")
            return np.array([[0., 0.],
                             [0., 0.]])
        if self.interp == 'pwc':
            try:
                i, j = int((nx - 1) * xx + 0.5), int((ny - 1) * yy + 0.5)
                if self.nt == 1:
                    return np.array([[self.d_u__d_x[0, i, j], self.d_u__d_y[0, i, j]],
                                     [self.d_v__d_x[0, i, j], self.d_v__d_y[0, i, j]]])
                else:
                    return (1 - alpha) * np.array([[self.d_u__d_x[it, i, j], self.d_u__d_y[it, i, j]],
                                                   [self.d_v__d_x[it, i, j], self.d_v__d_y[it, i, j]]]) + \
                           alpha * np.array([[self.d_u__d_x[it + 1, i, j], self.d_u__d_y[it + 1, i, j]],
                                             [self.d_v__d_x[it + 1, i, j], self.d_v__d_y[it + 1, i, j]]])
            except (ValueError, IndexError) as e:
                return np.array([[0., 0.],
                                 [0., 0.]])
        elif self.interp == 'linear':
            delta_x = (self.x_max - self.x_min) / (nx - 1)
            delta_y = (self.y_max - self.y_min) / (ny - 1)
            i = int((nx - 1) * xx)
            j = int((ny - 1) * yy)
            xij = delta_x * i + self.x_min
            a = (x[0] - xij) / delta_x
            yij = delta_y * j + self.y_min
            b = (x[1] - yij) / delta_y
            if self.nt == 1:
                d_uv__d_x = 1. / delta_x * ((1 - b) * (self.uv[0, i + 1, j] - self.uv[0, i, j]) +
                                            b * (self.uv[0, i + 1, j + 1] - self.uv[0, i, j + 1]))
                d_uv__d_y = 1. / delta_y * ((1 - a) * (self.uv[0, i, j + 1] - self.uv[0, i, j]) +
                                            a * (self.uv[0, i + 1, j + 1] - self.uv[0, i + 1, j]))
            else:
                d_uv__d_x = 1. / delta_x * ((1 - alpha) * ((1 - b) * (self.uv[it, i + 1, j] - self.uv[it, i, j]) +
                                                           b * (self.uv[it, i + 1, j + 1] - self.uv[it, i, j + 1])) +
                                            alpha * ((1 - b) * (self.uv[it + 1, i + 1, j] - self.uv[it + 1, i, j]) +
                                                     b * (self.uv[it + 1, i + 1, j + 1] - self.uv[it + 1, i, j + 1])))
                d_uv__d_y = 1. / delta_y * ((1 - alpha) * ((1 - a) * (self.uv[it, i, j + 1] - self.uv[it, i, j]) +
                                                           a * (self.uv[it, i + 1, j + 1] - self.uv[it, i + 1, j])) +
                                            alpha * ((1 - a) * (self.uv[it + 1, i, j + 1] - self.uv[it + 1, i, j]) +
                                                     a * (self.uv[it + 1, i + 1, j + 1] - self.uv[it + 1, i + 1, j])))
            return np.stack((d_uv__d_x, d_uv__d_y), axis=1)

    def compute_derivatives(self):
        """
        Computes the derivatives of the windfield with a central difference scheme
        on the wind native grid
        """
        if self.uv is None:
            raise RuntimeError("Derivative computation failed : wind not yet loaded")
        self.d_u__d_x = np.zeros((self.nt, self.nx - 2, self.ny - 2))
        self.d_u__d_y = np.zeros((self.nt, self.nx - 2, self.ny - 2))
        self.d_v__d_x = np.zeros((self.nt, self.nx - 2, self.ny - 2))
        self.d_v__d_y = np.zeros((self.nt, self.nx - 2, self.ny - 2))

        # Use order 2 precision derivative
        factor = 1 / EARTH_RADIUS if self.coords == 'gcs' else 1.
        self.d_u__d_x[:] = factor * 0.5 * (self.uv[:, 2:, 1:-1, 0] - self.uv[:, :-2, 1:-1, 0]) / (
                self.grid[2:, 1:-1, 0] - self.grid[:-2, 1:-1, 0])
        self.d_u__d_y[:] = factor * 0.5 * (self.uv[:, 1:-1, 2:, 0] - self.uv[:, 1:-1, :-2, 0]) / (
                self.grid[1:-1, 2:, 1] - self.grid[1:-1, :-2, 1])
        self.d_v__d_x[:] = factor * 0.5 * (self.uv[:, 2:, 1:-1, 1] - self.uv[:, :-2, 1:-1, 1]) / (
                self.grid[2:, 1:-1, 0] - self.grid[:-2, 1:-1, 0])
        self.d_v__d_y[:] = factor * 0.5 * (self.uv[:, 1:-1, 2:, 1] - self.uv[:, 1:-1, :-2, 1]) / (
                self.grid[1:-1, 2:, 1] - self.grid[1:-1, :-2, 1])

    def flatten(self, proj='plate-carree', **kwargs):
        """
        Flatten GCS-coordinates wind to euclidean space using given projection
        :return The flattened wind
        """
        if self.coords != COORD_GCS:
            print('Coordinates must be GCS to flatten', file=sys.stderr)
            exit(1)
        oldgrid = np.zeros(self.grid.shape)
        oldgrid[:] = self.grid
        oldbounds = (self.x_min, self.y_min, self.x_max, self.y_max)
        if proj == 'plate-carree':
            center = 0.5 * (self.grid[0, 0] + self.grid[-1, -1])
            newgrid = np.zeros(self.grid.shape)
            newgrid[:] = EARTH_RADIUS * (self.grid - center)
            self.grid[:] = newgrid
            self.x_min = self.grid[0, 0, 0]
            self.y_min = self.grid[0, 0, 1]
            self.x_max = self.grid[-1, -1, 0]
            self.y_max = self.grid[-1, -1, 1]
            f_wind = DiscreteWind()
            nx, ny, _ = self.grid.shape
            bl = self.grid[0, 0]
            tr = self.grid[-1, -1]
            f_wind.load_from_wind(self, nx, ny, bl, tr, coords=COORD_CARTESIAN)
        elif proj == 'ortho':
            for p in ['lon_0', 'lat_0']:
                if p not in kwargs.keys():
                    print(f'Missing parameter {p} for {proj} flatten type', file=sys.stderr)
                    exit(1)
            # Lon and lats expected in radians
            self.lon_0 = kwargs['lon_0']
            self.lat_0 = kwargs['lat_0']
            # width = kwargs['width']
            # height = kwargs['height']

            nx, ny, _ = self.grid.shape

            # Grid
            proj = Proj(proj='ortho', lon_0=RAD_TO_DEG * self.lon_0, lat_0=RAD_TO_DEG * self.lat_0)
            oldgrid = np.zeros(self.grid.shape)
            oldgrid[:] = self.grid
            newgrid = np.zeros((2, nx, ny))
            newgrid[:] = proj(RAD_TO_DEG * self.grid[:, :, 0], RAD_TO_DEG * self.grid[:, :, 1])
            self.grid[:] = newgrid.transpose((1, 2, 0))
            oldbounds = (self.x_min, self.y_min, self.x_max, self.y_max)
            self.x_min = self.grid[:, :, 0].min()
            self.y_min = self.grid[:, :, 1].min()
            self.x_max = self.grid[:, :, 0].max()
            self.y_max = self.grid[:, :, 1].max()

            print(self.x_min, self.x_max, self.y_min, self.y_max)

            # Wind
            newuv = np.zeros(self.uv.shape)
            for kt in range(self.uv.shape[0]):
                newuv[kt, :] = self._rotate_wind(self.uv[kt, :, :, 0],
                                                 self.uv[kt, :, :, 1],
                                                 RAD_TO_DEG * oldgrid[:, :, 0],
                                                 RAD_TO_DEG * oldgrid[:, :, 1])
            self.uv[:] = newuv
        else:
            print(f"Unknown projection type {proj}", file=sys.stderr)
            exit(1)

        # self.grid[:] = oldgrid
        # self.x_min, self.y_min, self.x_max, self.y_max = oldbounds

        self.unstructured = True
        self.coords = COORD_CARTESIAN
        self.units_grid = 'meters'
        self.is_analytical = False

    def _rotate_wind(self, u, v, x, y):
        if self.bm is None:
            self.bm = Basemap(projection='ortho', lon_0=RAD_TO_DEG * self.lon_0, lat_0=RAD_TO_DEG * self.lat_0)
        return np.array(self.bm.rotate_vector(u, v, x, y)).transpose((1, 2, 0))

    def dualize(self):
        # Override method so that the dual of a DiscreteWind stays a DiscreteWind and
        # is not casted to Wind
        wind = DiscreteWind()
        bl = (self.x_min, self.y_min)
        tr = (self.x_max, self.y_max)
        wind.load_from_wind(-1. * self, self.nx, self.ny, bl, tr, self.coords)
        if self.t_end is not None:
            wind.t_start = self.t_end
            wind.t_end = self.t_start
        else:
            wind.t_start = self.t_start
        return wind


class TwoSectorsWind(Wind):

    def __init__(self,
                 v_w1: float,
                 v_w2: float,
                 x_switch: float):
        """
        Wind configuration where wind is constant over two half-planes separated by x = x_switch. x-wind is null.

        :param v_w1: y-wind value for x < x_switch
        :param v_w2: y-wind value for x >= x_switch
        :param x_switch: x-coordinate for sectors separation
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.v_w1 = v_w1
        self.v_w2 = v_w2
        self.x_switch = x_switch
        self.descr = f'Two sectors'  # ({self.v_w1:.2f} m/s, {self.v_w2:.2f} m/s)'
        self.is_analytical = True

    def value(self, x):
        return np.array([0, self.v_w1 * np.heaviside(self.x_switch - x[0], 0.)
                         + self.v_w2 * np.heaviside(x[0] - self.x_switch, 0.)])

    def d_value(self, x):
        return np.array([[0, 0],
                         [0, 0]])


class TSEqualWind(TwoSectorsWind):

    def __init__(self, v_w1, v_w2, x_f):
        """
        TwoSectorsWind but the sector separation is midway to the target

        :param x_f: Target x-coordinate.
        """
        super().__init__(v_w1, v_w2, x_f / 2)


class UniformWind(Wind):

    def __init__(self, wind_vector: ndarray):
        """
        :param wind_vector: Direction and strength of wind
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.wind_vector = wind_vector
        self.descr = f'Uniform'  # ({np.linalg.norm(self.wind_vector):.2f} m/s, {math.floor(180 / np.pi * np.arctan2(self.wind_vector[1], self.wind_vector[0]))})'
        self.is_analytical = True

    def value(self, t, x):
        return self.wind_vector

    def d_value(self, t, x):
        return np.array([[0., 0.],
                         [0., 0.]])


class VortexWind(Wind):

    def __init__(self,
                 x_omega: float,
                 y_omega: float,
                 gamma: float):
        """
        A vortex from potential theory

        :param x_omega: x_coordinate of vortex center in m
        :param y_omega: y_coordinate of vortex center in m
        :param gamma: Circulation of the vortex in m^2/s. Positive is ccw vortex.
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.x_omega = x_omega
        self.y_omega = y_omega
        self.omega = np.array([x_omega, y_omega])
        self.gamma = gamma
        self.descr = f'Vortex'  # ({self.gamma:.2f} m^2/s, at ({self.x_omega:.2f}, {self.y_omega:.2f}))'
        self.is_analytical = True

    def value(self, t, x):
        r = np.linalg.norm(x - self.omega)
        e_theta = np.array([-(x - self.omega)[1] / r,
                            (x - self.omega)[0] / r])
        return self.gamma / (2 * np.pi * r) * e_theta

    def d_value(self, t, x):
        r = np.linalg.norm(x - self.omega)
        x_omega = self.x_omega
        y_omega = self.y_omega
        return self.gamma / (2 * np.pi * r ** 4) * \
               np.array([[2 * (x[0] - x_omega) * (x[1] - y_omega), (x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2],
                         [(x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2, -2 * (x[0] - x_omega) * (x[1] - y_omega)]])


class RankineVortexWind(Wind):

    def __init__(self, center, gamma, radius, t_start=0., t_end=None):
        """
        A vortex from potential theory

        :param center: Coordinates of the vortex center in m. Shape (2,) if steady else shape (nt, 2)
        :param gamma: Circulation of the vortex in m^2/s. Positive is ccw vortex. Scalar for steady vortex
        else shape (nt,)
        :param radius: Radius of the vortex core in meters. Scalar for steady vortex else shape (nt,)
        """
        time_varying = False
        nt = 1
        if len(center.shape) > 1:
            time_varying = True
            nt = center.shape[0]
        if time_varying and t_end is None:
            print('Missing t_end', file=sys.stderr)
            exit(1)
        super().__init__(value_func=self.value, d_value_func=self.d_value, t_start=t_start, t_end=t_end, nt=nt)

        self.omega = np.array(center)
        self.gamma = np.array(gamma)
        self.radius = np.array(radius)
        if time_varying and not (self.omega.shape[0] == self.gamma.shape[0] == self.radius.shape[0]):
            print('Incoherent sizes', file=sys.stderr)
            exit(1)
        self.zero_ceil = 1e-3
        self.descr = f'Rankine vortex'  # ({self.gamma:.2f} m^2/s, at ({self.x_omega:.2f}, {self.y_omega:.2f}))'
        self.is_analytical = True

    def max_speed(self, t=0.):
        _, gamma, radius = self.params(t)
        return gamma / (radius * 2 * np.pi)

    def params(self, t):
        if self.t_end is None:
            return self.omega, self.gamma, self.radius
        i, alpha = self._index(t)
        omega = (1 - alpha) * self.omega[i] + alpha * self.omega[i + 1]
        gamma = (1 - alpha) * self.gamma[i] + alpha * self.gamma[i + 1]
        radius = (1 - alpha) * self.radius[i] + alpha * self.radius[i + 1]
        return omega, gamma, radius

    def value(self, t, x):
        omega, gamma, radius = self.params(t)
        r = np.linalg.norm(x - omega)
        if r < self.zero_ceil * radius:
            return np.zeros(2)
        e_theta = np.array([-(x - omega)[1] / r,
                            (x - omega)[0] / r])
        f = gamma / (2 * np.pi)
        if r <= radius:
            return f * r / (radius ** 2) * e_theta
        else:
            return f * 1 / r * e_theta

    def d_value(self, t, x):
        omega, gamma, radius = self.params(t)
        r = np.linalg.norm(x - omega)
        if r < self.zero_ceil * radius:
            return np.zeros((2, 2))
        x_omega = omega[0]
        y_omega = omega[1]
        e_r = np.array([(x - omega)[0] / r,
                        (x - omega)[1] / r])
        e_theta = np.array([-(x - omega)[1] / r,
                            (x - omega)[0] / r])
        f = gamma / (2 * np.pi)
        if r <= radius:
            a = (x - omega)[0]
            b = (x - omega)[1]
            d_e_theta__d_x = np.array([a * b / r ** 3, b ** 2 / r ** 3])
            d_e_theta__d_y = np.array([- a ** 2 / r ** 3, - a * b / r ** 3])
            return f / radius ** 2 * np.stack((a / r * e_theta + r * d_e_theta__d_x,
                                               b / r * e_theta + r * d_e_theta__d_y), axis=1)
        else:
            return f / r ** 4 * \
                   np.array([[2 * (x[0] - x_omega) * (x[1] - y_omega), (x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2],
                             [(x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2, -2 * (x[0] - x_omega) * (x[1] - y_omega)]])


# class VortexBarrierWind(Wind):
#     def __init__(self,
#                  x_omega: float,
#                  y_omega: float,
#                  gamma: float,
#                  radius: float):
#         """
#         A vortex from potential theory, but add a wind barrier at a given radius
#
#         :param x_omega: x_coordinate of vortex center in m
#         :param y_omega: y_coordinate of vortex center in m
#         :param gamma: Circulation of the vortex in m^2/s. Positive is ccw vortex.
#         :param radius: Radius of the barrier in m
#         """
#         super().__init__(value_func=self.value, d_value_func=self.d_value)
#         self.x_omega = x_omega
#         self.y_omega = y_omega
#         self.omega = np.array([x_omega, y_omega])
#         self.gamma = gamma
#         self.radius = radius
#         self.zero_ceil = 1e-3
#         self.descr = f'Vortex barrier'  # ({self.gamma:.2f} m^2/s, at ({self.x_omega:.2f}, {self.y_omega:.2f}))'
#         self.is_analytical = True


class SourceWind(Wind):

    def __init__(self,
                 x_omega: float,
                 y_omega: float,
                 flux: float):
        """
        A source from potentiel theory

        :param x_omega: x_coordinate of source center in m
        :param y_omega: y_coordinate of source center in m
        :param flux: Flux of the source m^2/s. Positive is source, negative is well.
        """
        super().__init__()
        self.x_omega = x_omega
        self.y_omega = y_omega
        self.omega = np.array([x_omega, y_omega])
        self.flux = flux
        self.is_analytical = True

    def value(self, x):
        r = np.linalg.norm(x - self.omega)
        e_r = (x - self.omega) / r
        return self.flux / (2 * np.pi * r) * e_r

    def d_value(self, x):
        raise ValueError("No derivative implemented for source wind")


class LinearWind(Wind):
    """
    This wind is NOT a potential flow
    """

    def __init__(self, gradient: ndarray, origin: ndarray, value_origin: ndarray):
        """
        :param gradient: The wind gradient
        :param origin: The origin point for the origin value
        :param value_origin: The origin value
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)

        self.gradient = np.zeros(gradient.shape)
        self.origin = np.zeros(origin.shape)
        self.value_origin = np.zeros(value_origin.shape)

        self.gradient[:] = gradient
        self.origin[:] = origin
        self.value_origin[:] = value_origin
        self.is_analytical = True

    def value(self, t, x):
        return self.gradient.dot(x - self.origin) + self.value_origin

    def d_value(self, t, x):
        return self.gradient


class LinearWindT(Wind):

    def __init__(self, gradient, origin, value_origin, t_end):
        """
        :param gradient: The wind gradient, shape (nt, 2, 2)
        :param origin: The origin point for the origin value, shape (2,)
        :param value_origin: The origin value, shape (2,)
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value, t_end=t_end)

        self.gradient = np.zeros(gradient.shape)
        self.origin = np.zeros(origin.shape)
        self.value_origin = np.zeros(value_origin.shape)

        self.gradient[:] = gradient
        self.origin[:] = origin
        self.value_origin[:] = value_origin
        self.is_analytical = True
        self.nt = self.gradient.shape[0]

    def value(self, t, x):
        i, alpha = self._index(t)
        return ((1 - alpha) * self.gradient[i] + alpha * self.gradient[i + 1]) @ (x - self.origin) + self.value_origin

    def d_value(self, t, x):
        i, alpha = self._index(t)
        return (1 - alpha) * self.gradient[i] + alpha * self.gradient[i + 1]

    def dualize(self):
        return -1. * LinearWindT(self.gradient[::-1, :, :], self.origin, self.value_origin, self.t_end)


class PointSymWind(Wind):
    """
    From Techy 2011 (DOI 10.1007/s11370-011-0092-9)
    """

    def __init__(self, x_center: float, y_center: float, gamma: float, omega: float):
        """
        :param x_center: Center x-coordinate
        :param y_center: Center y-coordinate
        :param gamma: Divergence
        :param omega: Curl
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)

        self.center = np.array((x_center, y_center))

        self.gamma = gamma
        self.omega = omega
        self.mat = np.array([[gamma, -omega], [omega, gamma]])
        self.is_analytical = True

    def value(self, t, x):
        return self.mat @ (x - self.center)

    def d_value(self, t, x):
        return self.mat


class DoubleGyreWind(Wind):
    """
    From Li 2020 (DOI 10.1109/JOE.2019.2926822)
    """

    def __init__(self, x_center: float, y_center: float, x_wl: float, y_wl: float, ampl: float):
        """
        :param x_center: Bottom left x-coordinate
        :param y_center: Bottom left y-coordinate
        :param x_wl: x-axis wavelength
        :param y_wl: y-axis wavelength
        :param ampl: Amplitude in meters per second
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)

        self.center = np.array((x_center, y_center))

        # Phase gradient
        self.kx = 2 * pi / x_wl
        self.ky = 2 * pi / y_wl
        self.ampl = ampl
        self.is_analytical = True

    def value(self, t, x):
        xx = np.diag((self.kx, self.ky)) @ (x - self.center)
        return self.ampl * np.array((-sin(xx[0]) * cos(xx[1]), cos(xx[0]) * sin(xx[1])))

    def d_value(self, t, x):
        xx = np.diag((self.kx, self.ky)) @ (x - self.center)
        return self.ampl * np.array([[-self.kx * cos(xx[0]) * cos(xx[1]), self.ky * sin(xx[0]) * sin(xx[1])],
                                     [-self.kx * sin(xx[0]) * sin(xx[1]), self.ky * cos(xx[0]) * cos(xx[1])]])


class RadialGaussWind(Wind):
    def __init__(self, x_center: float, y_center: float, radius: float, sdev: float, v_max: float):
        """
        :param x_center: Center x-coordinate
        :param y_center: Center y-coordinate
        :param radius: Radial distance of maximum wind value (absolute value)
        :param sdev: Gaussian standard deviation
        :param v_max: Maximum wind value
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)

        self.center = np.array((x_center, y_center))

        self.radius = radius
        self.sdev = sdev
        self.v_max = v_max
        self.zero_ceil = 1e-3
        self.is_analytical = True
        self.dualizable = False

    def ampl(self, r):
        if r < self.zero_ceil * self.radius:
            return 0.
        return self.v_max * exp(-(log(r / self.radius)) ** 2 / (2 * self.sdev ** 2))

    def value(self, t, x):
        xx = x - self.center
        r = np.linalg.norm(xx)
        if r < self.zero_ceil * self.radius:
            return np.zeros(2)
        e_r = (x - self.center) / r
        v = self.ampl(r)
        return v * e_r

    def d_value(self, t, x):
        xx = x - self.center
        r = np.linalg.norm(xx)
        if r < self.zero_ceil * self.radius:
            return np.zeros((2, 2))
        e_r = (x - self.center) / r
        a = (x - self.center)[0]
        b = (x - self.center)[1]
        dv = -log(r / self.radius) * self.ampl(r) / (r ** 2 * self.sdev ** 2) * np.array([xx[0], xx[1]])
        nabla_e_r = np.array([[b ** 2 / r ** 3, - a * b / r ** 3],
                              [-a * b / r ** 3, a ** 2 / r ** 3]])
        return np.einsum('i,j->ij', e_r, dv) + self.ampl(r) * nabla_e_r


class RadialGaussWindT(Wind):
    """
    Time-varying version of radial Gauss wind
    """

    def __init__(self, center, radius, sdev, v_max, t_end, t_start=0.):
        """
        :param center: ndarray of size nt x 2
        :param radius: ndarray of size nt
        :param sdev: ndarray of size nt
        :param v_max: ndarray of size nt
        :param t_end: end of time window. Wind is supposed to be regularly sampled in the time window
        :param t_start: beginning of time window.
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value, t_start=t_start, t_end=t_end,
                         nt=radius.shape[0])

        self.center = np.zeros(center.shape)
        self.radius = np.zeros(radius.shape)
        self.sdev = np.zeros(sdev.shape)
        self.v_max = np.zeros(v_max.shape)

        self.center[:] = center
        self.radius[:] = radius
        self.sdev[:] = sdev
        self.v_max[:] = v_max

        self.zero_ceil = 1e-3
        self.is_analytical = True
        self.dualizable = False

    def ampl(self, t, r):
        i, alpha = self._index(t)
        radius = (1 - alpha) * self.radius[i] + alpha * self.radius[i + 1]
        v_max = (1 - alpha) * self.v_max[i] + alpha * self.v_max[i + 1]
        sdev = (1 - alpha) * self.sdev[i] + alpha * self.sdev[i + 1]
        if r < self.zero_ceil * radius:
            return 0.
        return v_max * exp(-(log(r / radius)) ** 2 / (2 * sdev ** 2))

    def value(self, t, x):
        i, alpha = self._index(t)
        center = (1 - alpha) * self.center[i] + alpha * self.center[i + 1]
        radius = (1 - alpha) * self.radius[i] + alpha * self.radius[i + 1]
        xx = x - center
        r = np.linalg.norm(xx)
        if r < self.zero_ceil * radius:
            return np.zeros(2)
        e_r = (x - center) / r
        v = self.ampl(t, r)
        return v * e_r

    def d_value(self, t, x):
        i, alpha = self._index(t)
        center = (1 - alpha) * self.center[i] + alpha * self.center[i + 1]
        radius = (1 - alpha) * self.radius[i] + alpha * self.radius[i + 1]
        sdev = (1 - alpha) * self.sdev[i] + alpha * self.sdev[i + 1]
        xx = x - center
        r = np.linalg.norm(xx)
        if r < self.zero_ceil * radius:
            return np.zeros((2, 2))
        e_r = (x - center) / r
        a = (x - center)[0]
        b = (x - center)[1]
        dv = -log(r / radius) * self.ampl(t, r) / (r ** 2 * sdev ** 2) * np.array([xx[0], xx[1]])
        nabla_e_r = np.array([[b ** 2 / r ** 3, - a * b / r ** 3],
                              [-a * b / r ** 3, a ** 2 / r ** 3]])
        return np.einsum('i,j->ij', e_r, dv) + self.ampl(t, r) * nabla_e_r


class BandGaussWind(Wind):
    """
    Linear band of gaussian wind
    """

    def __init__(self, origin, vect, ampl, sdev):
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.origin = np.zeros(2)
        self.origin[:] = origin
        self.vect = np.zeros(2)
        self.vect = vect / np.linalg.norm(vect)
        self.ampl = ampl
        self.sdev = sdev

    def value(self, t, x):
        dist = np.abs(np.cross(self.vect, x - self.origin))
        intensity = self.ampl * np.exp(-0.5 * (dist / self.sdev) ** 2)
        return self.vect * intensity

    def d_value(self, t, x):
        # TODO : write analytical formula
        dx = 1e-6
        return np.column_stack((1 / dx * (self.value(t, x + np.array((dx, 0.))) - self.value(t, x)),
                                1 / dx * (self.value(t, x + np.array((0., dx))) - self.value(t, x))))


class BandWind(Wind):
    """
    Band of wind. NON DIFFERENTIABLE (should be instanciated as discrete wind)
    """

    def __init__(self, origin, vect, w_value, width):
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.origin = np.zeros(2)
        self.origin[:] = origin
        self.vect = np.zeros(2)
        self.vect = vect / np.linalg.norm(vect)
        self.w_value = w_value
        self.width = width

    def value(self, t, x):
        dist = np.abs(np.cross(self.vect, x - self.origin))
        if dist >= self.width / 2.:
            return np.zeros(2)
        else:
            return self.w_value

    def d_value(self, t, x):
        print('Undefined', file=sys.stderr)
        exit(1)


class LCWind(Wind):
    """
    Linear combination of winds. Handles the "dualization" of the windfield, i.e. multiplying by -1 everything
    except wind virtual obstacles
    """

    def __init__(self, coeffs, winds):
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.coeffs = np.zeros(coeffs.shape[0])
        self.coeffs[:] = coeffs
        self.winds = winds

        self.lcwind = sum((c * self.winds[i] for i, c in enumerate(self.coeffs)), UniformWind(np.zeros(2)))
        nt = None
        t_end = None
        for wind in self.winds:
            if wind.nt > 1:
                if nt is None:
                    nt = wind.nt
                    t_end = wind.t_end
                else:
                    if wind.nt != nt or wind.t_end != t_end:
                        print('Cannot handle combination of multiple time-varying winds with'
                              ' different parameters for the moment', file=sys.stderr)
                        exit(1)
        self.nt = nt if nt is not None else 1
        self.t_end = t_end

    def value(self, t, x):
        return self.lcwind.value(t, x)

    def d_value(self, t, x):
        return self.lcwind.d_value(t, x)

    def dualize(self):
        new_coeffs = np.zeros(self.coeffs.shape[0])
        for i, c in enumerate(self.coeffs):
            if self.winds[i].dualizable:
                new_coeffs[i] = -1. * c
            else:
                new_coeffs[i] = c
        return LCWind(new_coeffs, self.winds)


class LVWind(Wind):
    """
    Wind varying linearly with time. Wind is spaitially uniform at each timestep.
    """

    def __init__(self, wind_value, gradient, time_scale):
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.wind_value = np.array(wind_value)
        self.gradient = np.array(gradient)
        self.t_end = time_scale

    def value(self, t, x):
        return self.wind_value + t * self.gradient

    def d_value(self, t, x):
        return np.zeros(2)


if __name__ == '__main__':
    wind = DiscreteWind(interp='linear')
    wind.load('/home/bastien/Documents/data/wind/windy/Dakar-Natal-0.5-padded.mz/data.h5')
    f_wind = wind.flatten()
