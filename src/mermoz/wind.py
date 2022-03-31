import os

import h5py
import numpy as np
from numpy import ndarray

from mermoz.misc import *


class Wind:

    def __init__(self, value_func=None, d_value_func=None, descr=''):
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
        self.is_dumpable = False

    def __add__(self, other):
        """
        Add windfields

        :param other: Another windfield
        :return: The sum of the two windfields
        """
        if not isinstance(other, Wind):
            raise TypeError(f"Unsupported type for addition : {type(other)}")
        return Wind(value_func=lambda x: self.value(x) + other.value(x),
                    d_value_func=lambda x: self.d_value(x) + other.d_value(x),
                    descr=self.descr + ', ' + other.descr)

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
        return Wind(value_func=lambda x: other * self.value(x),
                    d_value_func=lambda x: other * self.d_value(x),
                    descr=self.descr)

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

    def __init__(self):
        super().__init__(value_func=self.value, d_value_func=self.d_value)

        self.is_dumpable = True

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

        self.clipping_tol = 1e-2

    def load(self, filepath, nodiff=False):
        """
        Loads wind data from H5 wind data
        :param filepath: The H5 file contaning wind data
        :param nodiff: Do not compute wind derivatives
        """

        # Fill the wind data array
        print(f'{"Loading wind values...":<30}', end='')
        with h5py.File(filepath, 'r') as wind_data:
            self.coords = wind_data.attrs['coords']
            self.units_grid = wind_data.attrs['units_grid']

            # Checking consistency before loading
            ensure_coords(self.coords)
            ensure_units(self.units_grid)
            ensure_compatible(self.coords, self.units_grid)

            # Loading
            self.nt, self.nx, self.ny, _ = wind_data['data'].shape
            self.uv = np.zeros((self.nt, self.nx, self.ny, 2))
            self.uv[:, :, :, :] = wind_data['data']
            self.grid = np.zeros((self.nx, self.ny, 2))
            self.grid[:] = wind_data['grid']
            self.ts = np.zeros((self.nt,))
            self.ts[:] = wind_data['ts']

        # Post processing
        if self.units_grid == U_DEG:
            self.grid[:] = DEG_TO_RAD * self.grid

        self.x_min = np.max(self.grid[0, :, 0])
        self.x_max = np.min(self.grid[-1, :, 0])
        self.y_min = np.max(self.grid[:, 0, 1])
        self.y_max = np.min(self.grid[:, -1, 1])

        if not nodiff:
            self.compute_derivatives()

        print('Done')

    def load_from_wind(self, wind: Wind, nx, ny, bl, tr, coords, nodiff=False):
        self.coords = coords
        self.units_grid = 'meters'

        # Checking consistency before loading
        ensure_coords(self.coords)
        ensure_units(self.units_grid)
        ensure_compatible(self.coords, self.units_grid)

        self.x_min = bl[0]
        self.x_max = tr[0]
        self.y_min = bl[1]
        self.y_max = tr[1]

        self.nt, self.nx, self.ny = 1, nx, ny
        self.uv = np.zeros((self.nt, self.nx, self.ny, 2))
        self.grid = np.zeros((self.nx, self.ny, 2))
        delta_x = (self.x_max - self.x_min) / (self.nx - 1)
        delta_y = (self.y_max - self.y_min) / (self.ny - 1)
        for i in range(self.nx):
            for j in range(self.ny):
                point = np.array([self.x_min + i * delta_x, self.y_min + j * delta_y])
                self.uv[0, i, j, :] = wind.value(point)
                self.grid[i, j, :] = point

        self.ts = np.zeros((self.nt,))

        # Post processing
        # Loading derivatives
        if not nodiff:
            self.d_u__d_x = np.zeros((self.nt, self.nx - 2, self.ny - 2))
            self.d_u__d_y = np.zeros((self.nt, self.nx - 2, self.ny - 2))
            self.d_v__d_x = np.zeros((self.nt, self.nx - 2, self.ny - 2))
            self.d_v__d_y = np.zeros((self.nt, self.nx - 2, self.ny - 2))
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    point = np.array([self.x_min + i * delta_x, self.y_min + j * delta_y])
                    diff = wind.d_value(point)
                    self.d_u__d_x[0, i - 1, j - 1] = diff[0, 0]
                    self.d_u__d_y[0, i - 1, j - 1] = diff[0, 1]
                    self.d_v__d_x[0, i - 1, j - 1] = diff[1, 0]
                    self.d_v__d_y[0, i - 1, j - 1] = diff[1, 1]

    def value(self, x, units=U_METERS):
        """
        :param x: Point at which to give wind interpolation
        :param units: Units in which x is given
        :return: The interpolated wind value at x
        """
        # ensure_units(units)
        # ensure_compatible(self.coords, units)

        nx = self.nx
        ny = self.ny

        factor = 1.  # (DEG_TO_RAD if self.coords == COORD_GCS and units == U_DEG else 1.)

        xx = (factor * x[0] - self.x_min) / (self.x_max - self.x_min)
        yy = (factor * x[1] - self.y_min) / (self.y_max - self.y_min)

        eps = self.clipping_tol
        if xx < 0. - eps or xx > 1. + eps or yy < 0. - eps or yy > 1. + eps:
            # print(f"Real windfield undefined at ({x[0]:.3f}, {x[1]:.3f})")
            return np.array([0., 0.])

        return self.uv[0, int((nx - 1) * xx + 0.5), int((ny - 1) * yy + 0.5)]

    def d_value(self, x, units=U_METERS):
        """
        :param x: Point at which to give wind interpolation
        :param units: Units in which x is given
        :return: The interpolated wind value at x
        """
        # ensure_units(units)
        # ensure_compatible(self.coords, units)

        nx = self.nx
        ny = self.ny

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
        i, j = int((nx - 1) * xx + 0.5), int((ny - 1) * yy + 0.5)
        try:
            return np.array([[self.d_u__d_x[0, i, j], self.d_u__d_y[0, i, j]],
                             [self.d_v__d_x[0, i, j], self.d_v__d_y[0, i, j]]])
        except IndexError:
            return np.array([[0., 0.],
                             [0., 0.]])

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

    def value(self, x):
        return self.wind_vector

    def d_value(self, x):
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

    def value(self, x):
        r = np.linalg.norm(x - self.omega)
        e_theta = np.array([-(x - self.omega)[1] / r,
                            (x - self.omega)[0] / r])
        return self.gamma / (2 * np.pi * r) * e_theta

    def d_value(self, x):
        r = np.linalg.norm(x - self.omega)
        x_omega = self.x_omega
        y_omega = self.y_omega
        return self.gamma / (2 * np.pi * r ** 4) * \
               np.array([[2 * (x[0] - x_omega) * (x[1] - y_omega), (x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2],
                         [(x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2, -2 * (x[0] - x_omega) * (x[1] - y_omega)]])


class RankineVortexWind(Wind):

    def __init__(self,
                 x_omega: float,
                 y_omega: float,
                 gamma: float,
                 radius: float):
        """
        A vortex from potential theory

        :param x_omega: x_coordinate of vortex center in m
        :param y_omega: y_coordinate of vortex center in m
        :param gamma: Circulation of the vortex in m^2/s. Positive is ccw vortex.
        :param radius: Radius of the vortex core in meters.
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.x_omega = x_omega
        self.y_omega = y_omega
        self.omega = np.array([x_omega, y_omega])
        self.gamma = gamma
        self.radius = radius
        self.descr = f'Rankine vortex'  # ({self.gamma:.2f} m^2/s, at ({self.x_omega:.2f}, {self.y_omega:.2f}))'

    def max_speed(self):
        return self.gamma / (self.radius * 2 * np.pi)

    def value(self, x):
        r = np.linalg.norm(x - self.omega)
        e_theta = np.array([-(x - self.omega)[1] / r,
                            (x - self.omega)[0] / r])
        f = self.gamma / (2 * np.pi)
        if r <= self.radius:
            return f * r / (self.radius ** 2) * e_theta
        else:
            return f * 1 / r * e_theta

    def d_value(self, x):
        r = np.linalg.norm(x - self.omega)
        x_omega = self.x_omega
        y_omega = self.y_omega
        e_r = np.array([(x - self.omega)[0] / r,
                        (x - self.omega)[1] / r])
        e_theta = np.array([-(x - self.omega)[1] / r,
                            (x - self.omega)[0] / r])
        f = self.gamma / (2 * np.pi)
        if r <= self.radius:
            a = (x - self.omega)[0]
            b = (x - self.omega)[1]
            d_e_theta__d_x = np.array([a * b / r ** 3, (r ** 2 - a ** 2) / r ** 3])
            d_e_theta__d_y = np.array([(r ** 2 - b ** 2) / r ** 3, a * b / r ** 3])
            return f / self.radius ** 2 * np.stack((a / r * e_theta + r * d_e_theta__d_x,
                                                    b / r * e_theta + r * d_e_theta__d_y), axis=1)
        else:
            return f / r ** 4 * \
                   np.array([[2 * (x[0] - x_omega) * (x[1] - y_omega), (x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2],
                             [(x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2, -2 * (x[0] - x_omega) * (x[1] - y_omega)]])


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

    def value(self, x):
        return self.gradient.dot(x - self.origin) + self.value_origin

    def d_value(self, x):
        return self.gradient
