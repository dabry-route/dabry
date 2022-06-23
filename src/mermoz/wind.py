import scipy.interpolate as itp

import h5py
from mpl_toolkits.basemap import Basemap
from math import exp, log
from pyproj import Proj

from mermoz.misc import *


class Wind:

    def __init__(self, value_func=None, d_value_func=None, descr='', is_analytical=False, **kwargs):
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

        # To remember if an analytical wind was discretized for
        # display purposes.
        self.is_analytical = is_analytical

        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __add__(self, other):
        """
        Add windfields

        :param other: Another windfield
        :return: The sum of the two windfields
        """
        if not isinstance(other, Wind):
            raise TypeError(f"Unsupported type for addition : {type(other)}")
        value_func = lambda x: self.value(x) + other.value(x)
        d_value_func = lambda x: self.d_value(x) + other.d_value(x)
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
                    #**d)

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
        value_func = lambda x: other * self.value(x)
        d_value_func = lambda x: other * self.d_value(x)
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
                    is_analytical=is_analytical)
                    #**d)

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

    def __init__(self, interp='pwc', force_analytical=False):
        super().__init__(value_func=self.value, d_value_func=self.d_value)

        self.is_dumpable = 2

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

        # Either 'pwc' for piecewise constant wind or 'linear' for linear interpolation
        self.interp = interp

        self.clipping_tol = 1e-2

        self.is_analytical = force_analytical

    def load(self, filepath, nodiff=False, resample=1):
        """
        Loads wind data from H5 wind data
        :param filepath: The H5 file contaning wind data
        :param nodiff: Do not compute wind derivatives
        :param unstructured: True if grid is not evenly-spaced 2D grid (fixed dx and fixed dy)
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
                self.uv = np.zeros((self.nt, self.nx, self.ny, 2))
                self.uv[:, :, :, :] = wind_data['data']
                self.grid = np.zeros((self.nx, self.ny, 2))
                self.grid[:] = wind_data['grid']
                self.ts = np.zeros((self.nt,))
                self.ts[:] = wind_data['ts']
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
                                     self.grid.reshape((self.nx * self.ny, 2)), method='linear', fill_value=0.).reshape((self.nx, self.ny))
                v_itp = np.zeros((self.nx, self.ny))
                v_itp[:] = itp.griddata(grid_wind.reshape((nx_wind * ny_wind, 2)),
                                     v_wind.reshape((nx_wind * ny_wind,)),
                                     self.grid.reshape((self.nx * self.ny, 2)), method='linear', fill_value=0.).reshape((self.nx, self.ny))
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

    def load_from_wind(self, wind: Wind, nx, ny, bl, tr, coords, nodiff=False, skip_coord_check=False):
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

        xx = (x[0] - self.x_min) / (self.x_max - self.x_min)
        yy = (x[1] - self.y_min) / (self.y_max - self.y_min)

        eps = self.clipping_tol
        if xx < 0. - eps or xx >= 1. or yy < 0. - eps or yy >= 1.:
            # print(f"Real windfield undefined at ({x[0]:.3f}, {x[1]:.3f})")
            return np.array([0., 0.])

        if self.interp == 'pwc':
            try:
                return self.uv[0, int((nx - 1) * xx + 0.5), int((ny - 1) * yy + 0.5)]
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
            return (1 - b) * ((1 - a) * self.uv[0, i, j] + a * self.uv[0, i + 1, j]) + \
                   b * ((1 - a) * self.uv[0, i, j + 1] + a * self.uv[0, i + 1, j + 1])

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
        if self.interp == 'pwc':
            try:
                i, j = int((nx - 1) * xx + 0.5), int((ny - 1) * yy + 0.5)
                return np.array([[self.d_u__d_x[0, i, j], self.d_u__d_y[0, i, j]],
                                 [self.d_v__d_x[0, i, j], self.d_v__d_y[0, i, j]]])
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
            d_uv__d_x = 1. / delta_x * ((1 - b) * (self.uv[0, i + 1, j] - self.uv[0, i, j]) +
                                        b * (self.uv[0, i + 1, j + 1] - self.uv[0, i, j + 1]))
            d_uv__d_y = 1. / delta_y * ((1 - a) * (self.uv[0, i, j + 1] - self.uv[0, i, j]) +
                                        a * (self.uv[0, i + 1, j + 1] - self.uv[0, i + 1, j]))
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
                    print(f'Missing parameter {p} for {proj} flatten type', file=sys.stdin)
                    exit(1)
            self.lon_0 = kwargs['lon_0']
            self.lat_0 = kwargs['lat_0']
            # width = kwargs['width']
            # height = kwargs['height']

            nx, ny, _ = self.grid.shape

            # Grid
            proj = Proj(proj='ortho', lon_0=self.lon_0, lat_0=self.lat_0)
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
        bm = Basemap(projection='ortho', lon_0=self.lon_0, lat_0=self.lat_0)
        return np.array(bm.rotate_vector(u, v, x, y)).transpose((1, 2, 0))


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
        self.is_analytical = True

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
        self.zero_ceil = 1e-3
        self.descr = f'Rankine vortex'  # ({self.gamma:.2f} m^2/s, at ({self.x_omega:.2f}, {self.y_omega:.2f}))'
        self.is_analytical = True

    def max_speed(self):
        return self.gamma / (self.radius * 2 * np.pi)

    def value(self, x):
        r = np.linalg.norm(x - self.omega)
        if r < self.zero_ceil * self.radius:
            return np.zeros(2)
        e_theta = np.array([-(x - self.omega)[1] / r,
                            (x - self.omega)[0] / r])
        f = self.gamma / (2 * np.pi)
        if r <= self.radius:
            return f * r / (self.radius ** 2) * e_theta
        else:
            return f * 1 / r * e_theta

    def d_value(self, x):
        r = np.linalg.norm(x - self.omega)
        if r < self.zero_ceil * self.radius:
            return np.zeros((2, 2))
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
            d_e_theta__d_x = np.array([a * b / r ** 3, b ** 2 / r ** 3])
            d_e_theta__d_y = np.array([- a ** 2 / r ** 3, - a * b / r ** 3])
            return f / self.radius ** 2 * np.stack((a / r * e_theta + r * d_e_theta__d_x,
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

    def value(self, x):
        return self.gradient.dot(x - self.origin) + self.value_origin

    def d_value(self, x):
        return self.gradient


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

    def value(self, x):
        return self.mat @ (x - self.center)

    def d_value(self, x):
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

    def value(self, x):
        xx = np.diag((self.kx, self.ky)) @ (x - self.center)
        return self.ampl * np.array((-sin(xx[0]) * cos(xx[1]), cos(xx[0]) * sin(xx[1])))

    def d_value(self, x):
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

    def ampl(self, r):
        if r < self.zero_ceil * self.radius:
            return 0.
        return self.v_max * exp(-(log(r / self.radius)) ** 2 / (2 * self.sdev ** 2))

    def value(self, x):
        xx = x - self.center
        r = np.linalg.norm(xx)
        if r < self.zero_ceil * self.radius:
            return np.zeros(2)
        e_r = (x - self.center) / r
        v = self.ampl(r)
        return v * e_r

    def d_value(self, x):
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


if __name__ == '__main__':
    wind = DiscreteWind(interp='linear')
    wind.load('/home/bastien/Documents/data/wind/windy/Dakar-Natal-0.5-padded.mz/data.h5')
    f_wind = wind.flatten()
