import functools
import sys
import time
from datetime import datetime
from enum import Enum
from math import pi, acos, cos, sin, floor, atan2

import numpy as np
from numpy import ndarray

"""
misc.py
Defines various procedures for distance computation, time formatting,
cpu time measurement.

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


def non_terminal(func):
    func.terminal = False
    func.direction = -1.
    return func


def terminal(func):
    func.terminal = True
    func.direction = -1.
    return func


def _to_alpha(i: int):
    if i == 0:
        return ''
    else:
        return _to_alpha(i // 26) + chr(65 + (i % 26))


def to_alpha(i: int):
    if i == 0:
        return 'A'
    else:
        return _to_alpha(i)


def alpha_to_int(s: str):
    if s == '':
        return 0
    return ord(s[-1]) - 65 + 26 * alpha_to_int(s[:-1])


def diadic_valuation(i: int):
    if i % 2 == 1:
        return 0
    return 1 + diadic_valuation(i // 2)


def directional_timeopt_control(ff_val: ndarray, d: ndarray, srf_max: float):
    """
    :param ff_val: Flow field vector
    :param d: Direction vector (norm has no importance)
    :param srf_max: Maximum speed relative to ff
    :return: Control vector
    """
    _d = d / np.linalg.norm(d)
    n = np.array(((0., -1.), (1., 0.))) @ _d
    ff_ortho = ff_val @ n
    angle = np.arctan2(np.sqrt(np.max((srf_max ** 2 - ff_ortho ** 2, 0.))), ff_ortho)
    u = srf_max * np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))) @ (-n)
    return u


def is_possible_direction(ff_val: ndarray, d: ndarray, srf_max: float) -> bool:
    """
    :param ff_val: Flow field vector
    :param d: Direction vector (norm has no importance)
    :param srf_max: Maximum speed relative to ff
    :return: True if ground speed vector can align with d, False else
    """
    _d = d / np.linalg.norm(d)
    n = np.array(((0., -1.), (1., 0.))) @ _d
    ff_ortho = ff_val @ n
    return np.abs(ff_ortho) < srf_max


def csv_to_dict(csv_file_path):
    data_dict = {}
    with open(csv_file_path, 'r') as csvfile:
        import csv
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # skip header
        for row in csv_reader:
            key = row[0]
            if len(key) == 0:
                continue
            values = row[1:]
            data_dict[key] = dict(zip(header[1:], values))
    return data_dict


def triangle_mask_and_cost(x: ndarray, p1: ndarray, p2: ndarray, p3: ndarray,
                           c1: float, c2: float, c3: float) -> ndarray:
    """
    Builds a cost map as a linear interpolation of values (c1, c2, c3)
    over the triangle (p1, p2, p3) and masked to be zero out of the triangle
    :param x: Regular grid of space (nx, ny, 2)
    :param p1: First vertex coordinates (2,)
    :param p2: Second vertex coordinates (2,)
    :param p3: Third vertex coordinates (2,)
    :param c1: First vertex cost (2,)
    :param c2: Second vertex cost (2,)
    :param c3: Third vertex cost (2,)
    :return: (nx, ny) cost map
    """
    # Formulas from https://mathworld.wolfram.com/TriangleInterior.html
    v0 = p1
    v1 = p2 - p1
    v2 = p3 - p1
    infs = np.inf * np.ones(x.shape[:-1])
    if np.isclose(np.cross(v1, v2), 0.):
        return infs
    a = (np.cross(x, v2) - np.cross(v0, v2)) / np.cross(v1, v2)
    b = -(np.cross(x, v1) - np.cross(v0, v1)) / np.cross(v1, v2)
    return np.where(a > 0, np.where(b > 0, np.where(a + b < 1, ((1 - (a + b)) * c1 + a * c2 + b * c3),
                                                    infs), infs), infs)


class Coords(Enum):
    CARTESIAN = 'cartesian'
    GCS = 'gcs'

    @classmethod
    def from_string(cls, s: str):
        if s == 'cartesian':
            return Coords.CARTESIAN
        elif s == 'gcs':
            return Coords.GCS
        else:
            raise ValueError('Unknown coord type "%s"' % s)


class Units(Enum):
    METERS = 'meters'
    RADIANS = 'rad'
    DEGREES = 'degrees'

    @classmethod
    def from_string(cls, s: str):
        if s == 'meters':
            return Units.METERS
        elif s == 'rad':
            return Units.RADIANS
        elif s == 'degrees':
            return Units.DEGREES
        else:
            raise ValueError('Unknown units "%s"' % s)

class Utils:

    DEG_TO_RAD = pi / 180.
    RAD_TO_DEG = 180. / pi
    AIRSPEED_DEFAULT = 23.  # [m/s]

    TRAJ_PMP = 'pmp'
    TRAJ_INT = 'integral'
    TRAJ_PATH = 'path'
    TRAJ_OPTIMAL = 'optimal'

    EARTH_RADIUS = 6378.137e3  # [m] Earth equatorial radius

    @staticmethod
    def to_0_360(angle):
        """
        Bring angle to 0-360 domain
        :param angle: Angle in degrees
        :return: Same angle in 0-360
        """
        return angle - 360. * floor(angle / 360.)

    @staticmethod
    def to_m180_180(angle):
        angle = np.asarray(angle)
        return angle - 360. * np.floor((angle + 180) / 360.)

    @staticmethod
    def rectify(a, b):
        """
        Bring angles to 0-720 and keep order
        :param a: First angle in degrees
        :param b: Second angle in degrees
        :return: (a, b) but between 0 and 720 degrees
        """
        aa = Utils.to_0_360(a)
        bb = Utils.to_0_360(b)
        if aa > bb:
            bb += 360.
        return aa, bb

    @staticmethod
    def ang_principal(a):
        """
        Bring angle to ]-pi,pi]
        :param a: Angle in radians
        :return: Equivalent angle in the target interval
        """
        return np.angle(np.cos(a) + 1j * np.sin(a))

    @staticmethod
    def angular_diff(a1, a2):
        """
        The measure of real angular difference between a1 and a2
        :param a1: First angle in radians
        :param a2: Second angle in radians
        :return: Angular difference
        """
        b1 = Utils.ang_principal(a1)
        b2 = Utils.ang_principal(a2)
        if abs(b2 - b1) <= np.pi:
            return abs(b2 - b1)
        else:
            return 2 * np.pi - abs(b2 - b1)

    @staticmethod
    def linspace_sph(a1, a2, N):
        """
        Return a ndarray consisting of a linspace of angles between the two angles
        but distributed on the geodesic between angles on the circle (shortest arc)
        :param a1: First angle in radians
        :param a2: Second angle in radians
        :param N: Number of discretization points
        :return: ndarray of angles in radians
        """
        b1 = Utils.ang_principal(a1)
        b2 = Utils.ang_principal(a2)
        b_min = min(b1, b2)
        delta_b = Utils.angular_diff(b1, b2)
        if abs(b2 - b1) <= np.pi:
            return np.linspace(b_min, b_min + delta_b, N)
        else:
            return np.linspace(b_min - delta_b, b_min, N)

    @staticmethod
    def ensure_compatible(coords: Coords, units: Units):
        if coords == Coords.CARTESIAN:
            if units not in [Units.METERS]:
                raise ValueError(f'Uncompatible coords "{coords.value}" and grid units "{units.value}"')
        elif coords == Coords.GCS:
            if units not in [Units.RADIANS, Units.DEGREES]:
                raise ValueError(f'Uncompatible coords "{coords.value}" and grid units "{units.value}"')

    @staticmethod
    def central_angle(*args, mode='rad'):
        """
        Computes the central angle between given points
        :param args: May be given in (lon1, lat1, lon2, lat2)
        or in (ndarray(lon1, lat1), ndarray(lon2, lat2))
        :param mode: Whether lon/lat given in degrees or radians
        :return: Central angle in radians
        """
        if len(args) == 4:
            lon1 = args[0]
            lat1 = args[1]
            lon2 = args[2]
            lat2 = args[3]
        elif len(args) == 2:
            lon1 = args[0][0]
            lat1 = args[0][1]
            lon2 = args[1][0]
            lat2 = args[1][1]
        else:
            raise Exception('Incorrect argument format')

        factor = Utils.DEG_TO_RAD if mode == 'deg' else 1.
        phi1 = lon1 * factor
        phi2 = lon2 * factor
        lambda1 = lat1 * factor
        lambda2 = lat2 * factor
        delta_phi = abs(phi2 - phi1)
        delta_phi -= int(delta_phi / pi) * pi
        # def rectify(phi):
        #     return phi + 2*pi if phi <= pi else -2*pi if phi > pi else 0.
        # phi1 = rectify(phi1)
        # phi2 = rectify(phi2)
        tol = 1e-6
        arg = sin(lambda1) * sin(lambda2) + cos(lambda1) * cos(lambda2) * cos(delta_phi)
        if arg > 1. and arg - 1. < tol:
            return 0.
        if arg < -1. and -(arg + 1.) < tol:
            return pi * Utils.EARTH_RADIUS
        return arg

    @staticmethod
    def geodesic_distance(*args, mode='rad'):
        """
        Computes the great circle distance between two points on the earth surface
        :param args: May be given in (lon1, lat1, lon2, lat2)
        or in (ndarray(lon1, lat1), ndarray(lon2, lat2))
        :param mode: Whether lon/lat given in degrees or radians
        :return: Great circle distance in meters
        """
        c_angle = Utils.central_angle(*args, mode=mode)
        return acos(c_angle) * Utils.EARTH_RADIUS

    @staticmethod
    def proj_ortho(lon, lat, lon0, lat0):
        """
        :param lon: Longitude in radians
        :param lat: Latitude in radians
        :param lon0: Longitude of center in radians
        :param lat0: Latitude of center in radians
        :return: Projection in meters
        """
        return Utils.EARTH_RADIUS * np.array((np.cos(lat) * np.sin(lon - lon0),
                                              np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(
                                                  lon - lon0)))

    @staticmethod
    def proj_ortho_inv(x, y, lon0, lat0):
        rho = np.sqrt(x ** 2 + y ** 2)
        c = np.arcsin(rho / Utils.EARTH_RADIUS)
        return np.array((lon0 + np.arctan(
            x * np.sin(c) / (rho * np.cos(c) * np.cos(lat0) - y * np.sin(c) * np.sin(lat0))),
                         np.arcsin(np.cos(c) * np.sin(lat0) + y * np.sin(c) * np.cos(lat0) / rho)))

    @staticmethod
    def d_proj_ortho(lon, lat, lon0, lat0):
        return Utils.EARTH_RADIUS * np.array(
            ((np.cos(lat) * np.cos(lon - lon0), -np.sin(lat) * np.sin(lon - lon0)),
             (np.sin(lat0) * np.cos(lat) * sin(lon - lon0),
              np.cos(lat0) * np.cos(lat) + np.sin(lat0) * np.sin(lat) * np.cos(lon - lon0))))

    @staticmethod
    def d_proj_ortho_inv(x, y, lon0, lat0):
        lon, lat = tuple(Utils.proj_ortho_inv(x, y, lon0, lat0))
        det = Utils.EARTH_RADIUS ** 2 * (np.cos(lat0) * np.cos(lat) ** 2 * np.cos(lon - lon0) +
                                         np.sin(lat0) * np.sin(lat) * np.cos(lat))
        dp = Utils.d_proj_ortho(lon, lat, lon0, lat0)
        return 1 / det * np.array(((dp[1, 1], -dp[0, 1]), (-dp[1, 0], dp[0, 0])))

    @staticmethod
    def middle(x1, x2, coords: Coords):
        if coords == Coords.CARTESIAN:
            # x1, x2 shall be cartesian vectors in meters
            return 0.5 * (x1 + x2)
        else:
            # Assuming coords == COORD_GCS
            # x1, x2 shall be vectors (lon, lat) in radians
            bx = np.cos(x2[1]) * np.cos(x2[0] - x1[0])
            by = np.cos(x2[1]) * np.sin(x2[0] - x1[0])
            return x1[0] + atan2(by, np.cos(x1[1]) + bx), \
                   atan2(np.sin(x1[1]) + np.sin(x2[1]), np.sqrt((np.cos(x1[1]) + bx) ** 2 + by ** 2))

    @staticmethod
    def distance(x1, x2, coords: Coords):
        if coords == Coords.GCS:
            # x1, x2 shall be vectors (lon, lat) in radians
            return Utils.geodesic_distance(x1, x2)
        else:
            # Assuming coords == COORD_CARTESIAN
            # x1, x2 shall be cartesian vectors in meters
            return np.linalg.norm(x1 - x2)

    @staticmethod
    def decorate(ax, title=None, xlab=None, ylab=None, legend=None, xlim=None, ylim=None, min_yspan=None):
        ax.xaxis.grid(color='k', linestyle='-', linewidth=0.2)
        ax.yaxis.grid(color='k', linestyle='-', linewidth=0.2)
        ax.tick_params(direction='in')
        if xlab: ax.xaxis.set_label_text(xlab)
        if ylab: ax.yaxis.set_label_text(ylab)
        if title: ax.set_title(title, {'fontsize': 8})
        if legend is not None: ax.legend(legend, loc='best')
        if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
        if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
        if min_yspan is not None: Utils.ensure_yspan(ax, min_yspan)

    @staticmethod
    def ensure_yspan(ax, yspan):
        ymin, ymax = ax.get_ylim()
        if ymax - ymin < yspan:
            ym = (ymin + ymax) / 2
            ax.set_ylim(ym - yspan / 2, ym + yspan / 2)

    @staticmethod
    def enlarge(bl, tr, factor=1.1):
        """
        Take a bounding box defined by its corners and return new corners corresponding
        to the enlarged bounding box
        :param bl: Bottom left corner (ndarray (2,))
        :param tr: Top right corner (ndarray (2,))
        :param factor: Zoom factor
        :return: (bl, tr) of the enlarged bounding box
        """
        if type(bl) != ndarray:
            bl = np.array(bl)
        if type(tr) != ndarray:
            tr = np.array(tr)
        if tr[0] < bl[0] or tr[1] < bl[1]:
            print('Bounding box corners in wrong order', file=sys.stderr)
            exit(1)
        delta_x = tr[0] - bl[0]
        delta_y = tr[1] - bl[1]
        # Center
        c = 0.5 * (bl + tr)
        half_delta = 0.5 * np.array((delta_x, delta_y))
        return c - factor * half_delta, c + factor * half_delta

    @staticmethod
    def airspeed_opti(p, cost='dobrokhodov'):
        if cost == 'dobrokhodov':
            pn = np.linalg.norm(p)
            return Utils.airspeed_opti_(pn)
        elif cost == 'subramani':
            pn = np.linalg.norm(p)
            return pn / 2.

    @staticmethod
    def airspeed_opti_(pn):
        kp1 = 0.05
        kp2 = 1000
        return np.sqrt(pn / (6 * kp1) + np.sqrt(kp2 / (3 * kp1) + pn ** 2 / (36 * kp1 ** 2)))

    @staticmethod
    def power(airspeed, cost='dobrokhodov'):
        if cost == 'dobrokhodov':
            kp1 = 0.05
            kp2 = 1000
            return kp1 * airspeed ** 3 + kp2 / airspeed
        elif cost == 'subramani':
            return airspeed ** 2

    @staticmethod
    def linear_wind_alyt_traj(airspeed, gradient, x_init, x_target, theta_f=None):
        """
        Return the analytical solution to minimum time of travel between
        x_init and x_target in a constant wind gradient.
        x_init and x_target are assumed to lay on the x-axis
        Wind gradient is assumed to be cross-track.
        :param airspeed: The vehicle airspeed in meters per second
        :param gradient: The windfield gradient non-null component in inverse seconds
        :return: The analytical quickest trajectory
        """
        # Analytic optimal trajectory
        w = -gradient

        def analytic_traj(theta, theta_f):
            x = 0.5 * airspeed / w * (-1 / np.cos(theta_f) * (np.tan(theta_f) - np.tan(theta)) +
                                      np.tan(theta) * (1 / np.cos(theta_f) - 1 / np.cos(theta)) -
                                      np.log((np.tan(theta_f)
                                              + 1 / np.cos(theta_f)) / (
                                                     np.tan(theta) + 1 / np.cos(theta))))
            y = airspeed / w * (1 / np.cos(theta) - 1 / np.cos(theta_f))
            return x + x_target[0] - x_init[0], y

        def residual(theta_f):
            return analytic_traj(-theta_f, theta_f)[0] - x_init[0]

        if theta_f is None:
            import scipy.optimize
            theta_f = scipy.optimize.newton_krylov(residual, -np.pi / 2. + 1e-1)
            print(f'theta_f : {theta_f}')
        print(f'T : {2 / w * np.tan(theta_f)}')

        points = np.array(list(map(lambda theta: analytic_traj(theta, theta_f), np.linspace(-theta_f, theta_f, 1000))))
        from dabry.trajectory import Trajectory
        return Trajectory.cartesian(np.linspace(0, 1, points.shape[0]),  # Warning: Fictitious time parameterization !
                          points)

    @staticmethod
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    @staticmethod
    def has_intersec(a, b, c, d):
        return Utils.ccw(a, c, d) != Utils.ccw(b, c, d) and Utils.ccw(a, b, c) != Utils.ccw(a, b, d)

    @staticmethod
    def intersection(a, b, c, d):
        ref_dim = np.mean(list(map(np.linalg.norm, (a, b, c, d))))
        xdiff = (a[0] - b[0], c[0] - d[0])
        ydiff = (a[1] - b[1], c[1] - d[1])

        def det(u, v):
            return u[0] * v[1] - u[1] * v[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(a, b), det(c, d))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        if b[1] - a[1] < 1e-5 * ref_dim:
            if b[0] - a[0] < 1e-5 * ref_dim:
                t_ab = 0.
            else:
                t_ab = (x - a[0]) / (b[0] - a[0])
        else:
            t_ab = (y - a[1]) / (b[1] - a[1])
        if d[1] - c[1] < 1e-5 * ref_dim:
            if d[0] - c[0] < 1e-5 * ref_dim:
                t_cd = 0.
            else:
                t_cd = (x - c[0]) / (d[0] - c[0])
        else:
            t_cd = (y - c[1]) / (d[1] - c[1])
        return (x, y), t_ab, t_cd

    @staticmethod
    def time_fmt(duration):
        """
        Formats duration to proper units
        :param duration: Duration in seconds
        :return: String representation of duration in proper units
        """
        if duration < 200:
            return f'{duration:.2f}s'
        elif duration < 60 * 200:
            minutes = int(duration / 60)
            seconds = int(duration - 60 * int(duration / 60))
            return f'{minutes}m{seconds}s'
        else:
            hours = int(duration / 3600)
            minutes = int(60 * (duration / 3600 - hours))
            return f'{hours}h{minutes}m'

    @staticmethod
    def read_pb_params(s):
        return Utils.process_pb_params(*s.strip().split(' ')[:7])

    @staticmethod
    def process_pb_params(x_init_lon, x_init_lat, x_target_lon, x_target_lat, start_date, airspeed, altitude):
        x_init = np.array([Utils.to_m180_180(float(x_init_lon)), float(x_init_lat)])
        x_target = np.array([Utils.to_m180_180(float(x_target_lon)), float(x_target_lat)])
        airspeed = float(airspeed)

        st_d = start_date
        year = int(st_d[:4])
        aargs = [int(st_d[4 + 2 * i:4 + 2 * (i + 1)]) for i in range(4)]
        start_date = datetime(year, *aargs)
        level = str(int(altitude))
        return x_init, x_target, start_date, airspeed, level

    @staticmethod
    def in_lonlat_box(bl, tr, p):
        """
        Returns True if p is in the specified box
        :param bl: Bottom left corner (lon, lat) in radians
        :param tr: Top right corner (lon, lat) in radians
        :param p: Point to check in format (lon, lat) in radians
        :return: True if point in box, False else
        """
        bl_lon, tr_lon = Utils.rectify(Utils.RAD_TO_DEG * bl[0], Utils.RAD_TO_DEG * tr[0])
        p_lon = Utils.to_0_360(Utils.RAD_TO_DEG * p[0])
        return (bl_lon < p_lon < tr_lon or bl_lon < p_lon + 360 < tr_lon) and bl[1] < p[1] < tr[1]

    @staticmethod
    def interpolate(values, bl, spacings, state_ex, ndim_values_data=1):
        """
        Interpolates `values` (possibly multidimensional per node) defined over the grid at the given `state`.
        Adapted from hj_reachability package
        Clips to border for points outside boundaries
        """
        position = (state_ex - bl) / np.array(spacings)
        index_lo = np.floor(position).astype(np.int32)
        index_hi = index_lo + 1
        weight_hi = position - index_lo
        weight_lo = 1 - weight_hi
        index_lo, index_hi = tuple(
            np.clip(index, 0, np.array(values.shape[:-ndim_values_data])
                    - np.ones(values.ndim - ndim_values_data, dtype=np.int32))
            for index in (index_lo, index_hi))
        weight = functools.reduce(lambda x, y: x * y, np.ix_(*np.stack([weight_lo, weight_hi], -1)))
        return np.sum(
            weight[(...,) + (np.newaxis,) * (values.ndim - state_ex.shape[0])] *
            values[np.ix_(*np.stack([index_lo, index_hi], -1))], tuple(list(range(state_ex.shape[0]))))


class Chrono:
    def __init__(self, no_verbose=False):
        self.t_start = 0.
        self.t_end = 0.
        self.verbose = not no_verbose

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, msg=''):
        if self.verbose:
            print(f'[*] {msg}')
        self.t_start = time.time()

    def stop(self):
        self.t_end = time.time()
        diff = self.t_end - self.t_start
        if self.verbose:
            print(f'[*] Done ({self})')
        return diff

    def __str__(self):
        return Utils.time_fmt(self.t_end - self.t_start)


class Debug:

    def __init__(self):
        self.names = [
            'Yacine', 'Youssef', 'Abigail', 'Achille', 'Adélaïde', 'Adil', 'Alix', 'Amel', 'Amin',
            'Amina', 'Annabelle', 'Artémis', 'Athénaïs', 'Cameron', 'Castille', 'Céline', 'Charlie', 'Claudia',
            'Curtis',
            'Dan', 'Dana', 'Daphnée', 'Diane', 'El', 'Éléa', 'Elsa', 'Elvire', 'Elyes', 'Félicien',
            'Haroun', 'Hugo', 'Ilan', 'Iris', 'Isabella', 'Ismael', 'Jack', 'Jade', 'Julien', 'Leandro',
            'Léo', 'Lino', 'Livia', 'Luisa', 'Madeleine', 'Mariame', 'Mathys', 'Meryam', 'Naïla', 'Nicole',
            'Nino', 'Perla', 'Perrine', 'Ranim', 'Romain', 'Sam', 'Samuel', 'Sebastian', 'Serine', 'Solène',
            'Thalia', 'Tidiane', 'Tim', 'Toma', 'Valentina', 'Zakary', 'Aaliyah', 'Adam', 'Aden', 'Alice',
            'Allan', 'Ambroise', 'Amine', 'Anas', 'Antonin', 'Ariel', 'Aristide', 'Armel', 'Colette', 'Dana',
            'Daniella', 'Dany', 'Eden', 'Emmy', 'Enzo', 'Eugène', 'Eyal', 'Fatma', 'Filip', 'Gaston',
            'George', 'Hanaé', 'Hedi', 'Henri', 'Ilyan', 'India', 'Isaure', 'Jean-Baptiste', 'Jude', 'Kawtar',
            'Kenny', 'Kylian', 'Lana', 'Lara', 'Lassana', 'Laura', 'Layanah', 'Lila', 'Lise', 'Louise',
            'Lya', 'Malik', 'Malo', 'Margaux', 'Marin', 'Maxence', 'Maxime', 'Melissa', 'Menahem', 'Michelle',
            'Mouhamadou', 'Myla', 'Mylan', 'Nadine', 'Nélia', 'Neyla', 'Nolhan', 'Nour', 'Paola', 'Rafaël',
            'Rosa', 'Ryan', 'Sirine', 'Solène', 'Théodore', 'Warren', 'Yani', 'Yanis', 'Yasin', 'Abel',
            'Adel', 'Adrian', 'Alban', 'Alima', 'Amalia', 'Aminata', 'Amine', 'Anaelle', 'Ariane', 'Astrid',
            'Ayah', 'Ayman', 'Cheikh', 'Cyrus', 'Dylan', 'Elie', 'Éloïse', 'Emilie', 'Fares', 'Félicie',
            'Frédéric', 'Gautier', 'Hafsa', 'Hawa', 'Hocine', 'Ilian', 'Islem', 'Jason', 'Julian', 'Julie',
            'Kader', 'Kiara', 'Kiyan', 'Lahna', 'Lucia', 'Lyam', 'Lydia', 'Maïwenn', 'Malick', 'Marco',
            'Marnie', 'Méllina', 'Milica', 'Mohamed', 'Mohamed-Lamine', 'Ousmane', 'Paul', 'Pierre-Louis', 'Ranya',
            'Safiya',
            'Sam', 'Siham', 'Taha', 'Talia', 'Timothé', 'Tiphaine', 'Titouan', 'Valeria', 'Yanis', 'Abd',
            'Abdoul', 'Ahmed', 'Anaïs', 'Anas', 'Asma', 'Augustine', 'Carla', 'Cheikh', 'Dalia', 'Damien',
            'Daphné', 'Daria', 'Elisabeth', 'Elliot', 'Estelle', 'Éva', 'Evy', 'Gianni', 'Grace', 'Hadja',
            'Hugo', 'Isabella', 'Iyad', 'Iyed', 'Jade', 'Jayden', 'Kaïs', 'Kheira', 'Léo', 'Léonard',
            'Liv', 'Loïs', 'Ludivine', 'Lyah', 'Lylia', 'Mady', 'Maëlie', 'Mahault', 'Mateo', 'Mathias',
            'Matisse', 'Matthew', 'May', 'Michel', 'Noa', 'Nohé', 'Omar', 'Ornella', 'Oscar', 'Romaïssa',
            'Roméo', 'Samuel', 'Samy', 'Sasha', 'Sienna', 'Simone', 'Sophia', 'Timothé', 'Vera', 'Yannis',
            'Younes', 'Zack', 'Zadig', 'Zoé', 'Achille', 'Adriel', 'Amel', 'Andréa', 'Archibald', 'Armand',
            'Aurélien', 'Blanche', 'Camélia', 'Céline', 'Constant', 'Damien', 'Dan', 'Éden', 'Emmanuelle', 'Eulalie',
            'Ève', 'Fadi', 'Geneviève', 'Gianni', 'Grégoire', 'Helena', 'Ilian', 'Issa', 'Joseph', 'Judith',
            'Julia', 'Kadidia', 'Kaïs', 'Karim', 'Lana', 'Léandre', 'Léonard', 'Linda', 'Loïs', 'Lorenzo',
            'Ludivine', 'Lyne', 'Maïlys', 'Malick', 'Malik', 'Maryam', 'Mateo', 'Mathis', 'Matisse', 'Mayline',
            'Mélissa', 'Melvil', 'Muhammad', 'Nahyl', 'Naïl', 'Neïla', 'Noah', 'Nour', 'Pierre', 'Prune',
            'Rafaela', 'Raphaël', 'Ryad', 'Sarah', 'Sira', 'Soan', 'Tal', 'Timothé', 'Tristan', 'Viktor',
            'Yacoub', 'Zéphyr', 'Adonis', 'Adrien', 'Aïcha', 'Alana', 'Aliyah', 'Alyah', 'Ambroise', 'Amira',
            'Anaëlle', 'Andy', 'Armance', 'Augustin', 'Awa', 'Aymen', 'Balthazar', 'Billie', 'Charline', 'Cheikh',
            'Dahlia', 'Eléa', 'Elisa', 'Elyes', 'Ernest', 'Ethel', 'Eva', 'Hadrien', 'Haron', 'Helena',
            'Issa', 'Jayden', 'Jérémie', 'Joyce', 'Julien', 'Laure', 'Laurine', 'Lazar', 'Liya', 'Luka',
            'Madeleine', 'Mahaut', 'Malik', 'Marcello', 'Margaux', 'Mariam', 'Marion', 'Mathias', 'Matthieu', 'Nana',
            'Nassim', 'Nayel', 'Nelia', 'Nicole', 'Nourane', 'Raphael', 'Salimata', 'Sidney', 'Soren', 'Souleymane',
            'Thelma', 'Ulysse', 'Violette', 'Wael', 'Yann', 'Zakarya', 'Abdoul-Aziz', 'Aimée', 'Albin', 'Angèle',
            'Anir', 'Asma', 'Audrey', 'Awa', 'Benoît', 'Castille', 'Constance', 'Dania', 'Eve', 'Fatou',
            'Hamza', 'Hussein', 'Imran', 'Janna', 'Jean-Baptiste', 'Jessie', 'Layane', 'Léonie', 'Lola', 'Lukas',
            'Mamadou', 'Marianne', 'Mark', 'Matei', 'Mathieu', 'Matthias', 'Mayeul', 'Meyron', 'Mylan', 'Nabil',
            'Nassim', 'Oscar', 'Paulin', 'Rita', 'Robinson', 'Roméo', 'Sébastien', 'Selma', 'Tara', 'Theodore',
            'Victor', 'Vladimir', 'Yuna', 'Arthur', 'Alexandre', 'Nina', 'Victoire', 'Thomas', 'Céleste', 'Ayden',
            'Eden', 'Robin', 'Albane', 'Blanche', 'Daphné', 'Assia', 'Noa', 'Faustine', 'Louisa', 'Roman',
            'Billie', 'Jacob', 'Marc', 'Théophile', 'Thibault', 'Oumou', 'Stella', 'Swann', 'Mona', 'Mahamadou',
            'Dania', 'Sekou', 'Leila', 'Mouhamadou', 'Halima', 'Mathieu', 'Layana', 'Ewen', 'Guillaume', 'Bilel',
            'Emilio', 'Angélina', 'Antonio', 'Axel', 'Bettina', 'Brune', 'Calixte', 'Capucine', 'Cécilia', 'Charline',
            'Cheick', 'Demba', 'Denis', 'Dora', 'Eden', 'Fanny', 'Fantine', 'Guillaume', 'Idrissa', 'Ilana',
            'Iliana', 'Jacques', 'Jenny', 'Joyce', 'Junior', 'Karim', 'Kenny', 'Kylian', 'Lauriane', 'Leila',
            'Léopoldine', 'Lili', 'Lily', 'Louis', 'Lucien', 'Luna', 'Manon', 'Mariama', 'Marlon', 'Matisse',
            'Perrine', 'Rayan', 'Réphaël', 'Sarah', 'Stéphanie', 'Thelma', 'Théophile', 'Thibaud', 'Tiphaine', 'Warren',
            'Yoni', 'Yves', 'Abdelkader', 'Abel', 'Aline', 'Anatole', 'Antony', 'Aristide', 'Assa', 'Auguste',
            'Aylan', 'Bianca', 'Brune', 'Cameron', 'Carmen', 'Chris', 'Clarence', 'Demba', 'Deniz', 'Edouard',
            'Ella', 'Émeline', 'Emmy', 'Énora', 'Éric', 'Gaëtan', 'Grâce', 'Grégory', 'Hannah', 'Ilhan',
            'Ilian', 'Isidore', 'Joachim', 'Joey', 'Kelvin', 'Lana', 'Lauryn', 'Lenny', 'Lilian', 'Linda',
            'Lohan', 'Louison', 'Lucas', 'Maeva', 'Maëva', 'Malik', 'Mickaël', 'Miguel', 'Néo', 'Noha',
            'Ondine', 'Paulin', 'Ranya', 'Richard', 'Riyad', 'Sabrina', 'Sacha', 'Safia', 'Sami', 'Serena',
            'Sofia', 'Taly', 'Tomas', 'Wendy', 'Youssef', 'Aaliyah', 'Albane', 'Alexandre', 'Alice', 'Amira',
            'Anaelle', 'Astrid', 'Aubin', 'Audrey', 'Camila', 'Carla', 'Célestine', 'Celina', 'Cheyenne', 'Claire',
            'Cyril', 'Damian', 'Daphné', 'Elliot', 'Emilia', 'Emmanuelle', 'Feriel', 'Filip', 'Frédéric', 'Grâce',
            'Hortense', 'Ianis', 'Ibrahim', 'Ilyes', 'Inès', 'Isabelle', 'Issam', 'Ivan', 'Jane', 'Julian',
            'Kilyan', 'Kylian', 'Leny', 'Léopoldine', 'Lise', 'Luca', 'Mahamadou', 'Matthieu', 'Merwan', 'Michel',
            'Nazim', 'Noha', 'Pauline', 'Philéas', 'Pierre-Louis', 'Rachel', 'Rayane', 'Reda', 'Sabrina', 'Shaïna',
            'Sidonie', 'Solène', 'Suzanne', 'Tidiane', 'Vianney', 'Wael', 'Warren', 'Wendy', 'Xavier', 'Ysée',
            'Aaron', 'Abdoul', 'Adama', 'Ahmad', 'Aïcha', 'Alessio', 'Alfred', 'Ambrine', 'Anastasia', 'Anya',
            'Baptiste', 'Blanche', 'Capucine', 'Célestine', 'Charlize', 'Cléa', 'Clothilde', 'Corto', 'Cyrine', 'Éléa',
            'Eléonore', 'Elise', 'Ewen', 'François', 'Hamza', 'Haya', 'Idan', 'Isée', 'Jad', 'Kamélia',
            'Karl', 'Kawtar', 'Keren', 'Kévin', 'Laurent', 'Leonardo', 'Loïs', 'Louka', 'Maïwenn', 'Maryam',
            'Mateja', 'Matteo', 'Maxence', 'Mayane', 'Mayeul', 'Melvil', 'Mickael', 'Milan', 'Nathalie', 'Nolann',
            'Nora', 'Raphael', 'Rayan', 'Roland', 'Romane', 'Sacha', 'Selyan', 'Shaïly', 'Solange', 'Stanislas',
            'Ugo', 'Victoire', 'Wandrille', 'Yaïr', 'Yanis', 'Younes', 'Abdel', 'Abigaël', 'Adame', 'Alban',
            'Alison', 'Alizée', 'Amira', 'Andy', 'Apolline', 'Ary', 'Baudouin', 'Bilel', 'Calista', 'Camelia',
            'Chaï', 'Christian', 'Clémentine', 'Djibril', 'Eliott', 'Eva', 'Fanta', 'Farah', 'Félicie', 'Fleur',
            'Franck', 'Gabriel', 'Giovanni', 'Grégory', 'Guy', 'Harold', 'Iban', 'Selma', 'Sharon', 'Shirel',
            'Thomas', 'Vadim', 'Wilson', 'Yannis', 'Younès', 'Alon', 'Amara', 'Ana', 'Angélina', 'Anne-Sophie',
            'Auriane', 'Ayman', 'Baptiste', 'Cléo', 'Dalia', 'Djénéba', 'Elisabeth', 'Emmanuel', 'Emmy', 'Eulalie',
            'Fanta', 'Gaspard', 'Georges', 'Hana', 'Hector', 'Ilyes', 'Isis', 'Ismaël', 'Jason', 'Joan',
            'Justine', 'Kahina', 'Karen', 'Karima', 'Kilian', 'Kylian', 'Lilas', 'Louison', 'Lynn', 'Mahé',
            'Manuel', 'Marouane', 'Mathis', 'Matthieu', 'Mayeul', 'Melvin', 'Michel', 'Moustapha', 'Noor', 'Oren',
            'Pierre-Antoine', 'Rodrigue', 'Roméo', 'Tancrède', 'Tanguy', 'Tao', 'Thaïs', 'Thelma', 'Thomas', 'Titouan',
            'Tobias', 'Yacouba', 'Yann', 'Yaya', 'Yohan', 'Achille', 'Alain', 'Alvin', 'Amanda', 'Irina',
            'Isis', 'Jarod', 'Jérémie', 'Joachim', 'John', 'Jonathan', 'Josh', 'Juliette', 'Kenan', 'Kevin',
            'Khadidja', 'Kylian', 'Leonardo', 'Livia', 'Lola', 'Lorette', 'Louna', 'Marcel', 'Maryline', 'Mateo',
            'Merlin', 'Moussa', 'Natacha', 'Nawel', 'Olga', 'Oren', 'Oussama', 'Paco', 'Rafaël', 'Ramy',
            'Rebecca', 'Roman', 'Sabrina', 'Salimata', 'Angèle', 'Anna', 'Asma', 'Brune', 'Bruno', 'Camila',
            'Celeste', 'Claire', 'Cléa', 'Dounia', 'Eden', 'Ely', 'Emmy', 'Firas', 'Gabriela', 'Georges',
            'Grégoire', 'Hélie', 'Hiba', 'Isaac', 'Jacob', 'Jimmy', 'Joris', 'Julia', 'Lahna', 'Lauren',
            'Léandre', 'Léane', 'Leyna', 'Lise', 'Liv', 'Loane', 'Louanne', 'Loup', 'Luan', 'Lucie',
            'Luke', 'Lylia', 'Lyna', 'Macéo', 'Maïssa', 'Maja', 'Maksim', 'Malo', 'Marc', 'Maya',
            'Méline', 'Mya', 'Naomie', 'Nell', 'Nelson', 'Nizar', 'Raphaëlle', 'Riad', 'Salomé', 'Sandro',
            'Séléna', 'Shirel', 'Tasnime', 'Warren', 'Adame', 'Alexander', 'Amelia', 'Anselme', 'Arman', 'Artus',
            'Basile', 'Ben', 'Bianca', 'Boubacar', 'Celia', 'César', 'Charly', 'Clémence', 'Cyprien', 'Dalla',
            'Diana', 'Dina', 'Elie', 'Emmy', 'Erwan', 'Eve', 'Ewan', 'Félicie', 'Gaëlle', 'Garance',
            'Gustave', 'Hadrien', 'Héloïse', 'Ibrahima', 'Jean-Baptiste', 'Jeremy', 'Joachim', 'Joan', 'Juliana',
            'Kadiatou',
            'Kenzi', 'Khalil', 'Kim', 'Liana', 'Lila', 'Lily', 'Loïse', 'Lorenzo', 'Lucie', 'Maïssa',
            'Marko', 'Maryam', 'Mohamed-Amine', 'Mona', 'Myriam', 'Neïla', 'Nelly', 'Nolhan', 'Olympe', 'Oumou',
            'Roméo', 'Rosa', 'Roxane', 'Saad', 'Samba', 'Sasha', 'Tanya', 'Thelma', 'Théophane', 'Tiana',
            'Johanna', 'Khadidja', 'Lucy', 'Aliou', 'Hamidou', 'Lukas', 'Priam', 'Alizée', 'Camilla', 'Malaïka',
            'Rayhana', 'Zohra', 'Jeanne', 'Alexandre', 'Ava', 'Nina', 'Noé', 'Basile', 'Jade', 'Martin',
            'Giulia', 'Manon', 'Lucie', 'Pablo', 'Ezra', 'Marcel', 'Aïcha', 'Elena', 'Elie', 'Olympe',
            'Aurèle', 'Jacques', 'Hamza', 'Pénélope', 'Zélie', 'Judith', 'Amina', 'Lena', 'Astrid', 'Jana',
            'Lenny', 'Romain', 'Kamil', 'Khalil', 'Daouda', 'Swann', 'Elisabeth', 'Jannah', 'Julien', 'Sekou',
            'Nola', 'Khadidja', 'Nessa', 'Wassim', 'Amy', 'Waël', 'Adama', 'Coumba', 'Ilian', 'Marco',
            'Joanne', 'Sana', 'Eléna', 'Eya', 'Selena', 'Aden', 'Alfred', 'Kenzo', 'Mady', 'Mylan',
            'Selim', 'Aby', 'Aïssatou', 'Dalia', 'Elissa', 'Enora', 'Mayline', 'Sakina', 'Akram', 'Elyo',
            'Kenan', 'Kenzi', 'Leon', 'Olivier', 'Tyler', 'Elsie', 'Linda', 'Milena', 'Olympia', 'Masten',
            'Tony', 'Assil', 'Ayna', 'Béatrice', 'Hajar', 'Marta', 'Zeinab', 'Elhadj', 'Eloi', 'Timothy',
            'Warren', 'Ayla', 'Nell', 'Sélène', 'Suzie', 'Aliocha', 'Kyan', 'Avigaïl', 'Cara', 'Elyna',
            'Layel', 'Maïssane', 'Adam', 'Louise', 'Basile', 'Auguste', 'Lucie', 'Madeleine', 'Youssef', 'Éléonore',
            'Elio', 'Aurèle', 'Kaïs', 'Elisa', 'Hannah', 'Jannah', 'Kylian', 'Gaston', 'Nael', 'Ana',
            'Sara', 'Emilie', 'Aboubacar', 'Isaïah', 'Anastasia', 'Esmée', 'Maïmouna', 'Zakariya', 'Anne', 'Zaynab',
            'Yahya', 'Andrea', 'Alassane', 'Kamil', 'Isis', 'Alphonse', 'Marlon', 'Fatim', 'Khady', 'Maëva',
            'Naïa', 'Simone', 'Arthus', 'Elyo', 'Foucauld', 'Mylann', 'Vasco', 'Wael', 'Waël', 'Leïa',
            'Maram', 'Mira', 'Razane', 'Jamal', 'Mayeul', 'Safwan', 'Xavier', 'Janelle', 'Coumba', 'Lya',
            'Théodora', 'Philippe', 'Zéphyr', 'Laetitia', 'Stefan', 'Binta', 'Clotilde', 'France', 'Alhassane', 'Kenan',
            'Khalifa', 'Levi', 'Orion', 'Akshaya', 'Lalya', 'Rosa', 'Sadio', 'Tali', 'Gabriel', 'Lina',
            'Julia', 'Malo', 'Clara', 'Théo', 'Jean', 'Elsa', 'Alexis', 'Ezra', 'Lila', 'Arsène',
            'Lily', 'Rafael', 'Léonore', 'Maëlle', 'Antonin', 'Tristan', 'Gaïa', 'Noor', 'Andréa', 'Lyne',
            'Zacharie', 'Alexandra', 'Mina', 'Élisa', 'Eloïse', 'Imany', 'Salimata', 'Aris', 'Timéo', 'Assiya',
            'Dania', 'Marguerite', 'Milann', 'Logan', 'Viktor', 'Yusuf', 'Anaé', 'Kayla', 'Mayssa', 'Hassan',
            'Sekou', 'Ada', 'Haya', 'Archibald', 'Élias', 'François', 'Philippa', 'Vanessa', 'Arié', 'Fares',
            'Inayah', 'Joanna', 'Selena', 'Ahmad', 'Aydan', 'Jibril', 'Oliver', 'Cécile', 'Jannat', 'Aubin',
            'Bradley', 'Loïc', 'Soën', 'Isabelle', 'Izïa', 'Neïla', 'Nihel', 'Scarlett', 'Maho', 'Shahine',
            'Anita', 'Beatrice', 'Giorgia', 'Lyra', 'Maëline', 'Mayssane', 'Séraphine', 'Yana', 'Chloé', 'Iris',
            'Charlotte', 'Léonard', 'Nour', 'Clémence', 'Stella', 'Issa', 'Edgar', 'Balthazar', 'Mona', 'Ariane',
            'Ines', 'Lilia', 'Aboubacar', 'Owen', 'Andrea', 'Isaiah', 'Roman', 'Antonin', 'Mélina', 'Irène',
            'Aris', 'Ishaq', 'Marguerite', 'Lehna', 'Rayane', 'Binta', 'Charline', 'Mayar', 'Aloïs', 'Boubacar',
            'Mathieu', 'Cléa', 'Bastien', 'Léana', 'Marwa', 'Philippe', 'Tidiane', 'Titouan', 'Israa', 'Lucy',
            'Naïla', 'Thalie', 'Jonathan', 'Milhan', 'Céline', 'Tamara', 'Florent', 'Paulin', 'Youcef', 'Aleyna',
            'Isée', 'Kadidia', 'Laure', 'Maïna', 'Safiatou', 'Zeinab', 'Malone', 'Oren', 'Elyne', 'Jemima',
            'Leïna', 'Marilou', 'Salima', 'Manil', 'Muhammed', 'Thierno', 'Wael', 'Zélie', 'Adama', 'Adama',
            'Adame', 'Ali', 'Alima', 'Amaël', 'Anaïs', 'Anastasia', 'Angela', 'Anis', 'Artus', 'Assetou',
            'Aubin', 'Baya', 'Bilel', 'Bintou', 'Camil', 'Caroline', 'Célestine', 'Cheikh', 'Chloé', 'Danny',
            'Delphine', 'Dimitri', 'Elina', 'Eloan', 'Elyssa', 'Emile', 'Emmy', 'Fabio', 'Fady', 'Fanta',
            'Fiona', 'Frida', 'Gabriella', 'Giulia', 'Haroune', 'Hayden', 'Isaure', 'Joaquim', 'Léna', 'Liam',
            'Liyah', 'Liza', 'Lorenzo', 'Lou', 'Louka', 'Loup', 'Luce', 'Lyna', 'Maël', 'Mahaut',
            'Maïlys', 'Maïna', 'Manelle', 'Manon', 'Margot', 'Marina', 'Marwa', 'Marwan', 'Michael', 'Milla',
            'Nadir', 'Nora', 'Nour', 'Noura', 'Paul-Arthur', 'Pierre', 'Sadio', 'Samir', 'Shaïna', 'Théophile',
            'Tiffany', 'Tiguida', 'Tina', 'Tybalt', 'Virgile', 'Ysée', 'Abdramane', 'Aksel', 'Alba', 'Albane',
            'Alix', 'Amandine', 'Anatole', 'Anissa', 'Antoine', 'Archibald', 'Augustin', 'Ava', 'Ayaan', 'Ayoub',
            'Azad', 'Basile', 'Bérénice', 'Camille', 'Charles', 'Chloé', 'Constance', 'Corentin', 'Diane', 'Ellie',
            'Elliot', 'Elya', 'Emilie', 'Emmanuelle', 'Eva', 'Éva', 'Héléna', 'Hélèna', 'Ines', 'Isis',
            'Jaden', 'Lilian', 'Lilwenn', 'Lisandro', 'Lison', 'Lou-Ann', 'Louison', 'Louka', 'Lucia', 'Lucie',
            'Lucille', 'Lukas', 'Lyes', 'Marie', 'Marilou', 'Marine', 'Matthias', 'Max', 'Mayane', 'Moussa',
            'Nada', 'Nicolas', 'Nicole', 'Oumou', 'Raphael', 'Richard', 'Rodrigo', 'Roxane', 'Sean', 'Séréna',
            'Sofia', 'Vanessa', 'Wilson', 'Younes', 'Youssef', 'Amani', 'Bastien', 'Castille', 'Cléa', 'Clémentine',
            'Colin', 'Dan', 'Deva', 'Divine', 'Éliott', 'Elise', 'Elyssa', 'Emilio', 'Eric', 'Ezio',
            'Fanta', 'Gaétan', 'Giacomo', 'Giulian', 'Hannah', 'Inaya', 'Ismail', 'Judith', 'Kendra', 'Khaled',
            'Layanah', 'Lise', 'Louisa', 'Lylia', 'Lyna', 'Marceau', 'Martin', 'Mathis', 'Mélanie', 'Milan',
            'Moussa', 'Nadine', 'Orso', 'Paola', 'Paulin', 'Rita', 'Salima', 'Salimata', 'Salomé', 'Sarah',
            'Serena', 'Sibylle', 'Solal', 'Stella', 'Swann', 'Thelma', 'Tidiane', 'Tina', 'Ysé', 'Zohra',
            'Aksel', 'Albéric', 'Albert', 'Amine', 'Ange', 'Aurore', 'Ayline', 'Bianca', 'Bilal', 'Bilel',
            'Cameron', 'Cassandra', 'Cassandre', 'Célia', 'Charles', 'Christ', 'Clarence', 'Coline', 'Elouan', 'Ema',
            'Emna', 'Eulalie', 'Fatimata', 'Firdaws', 'Fleur', 'Gaspard', 'Gina', 'Giulia', 'Héléna', 'Ishaq',
            'Ismaël', 'Ismail', 'Jacob', 'Jennah', 'Juan', 'Judith', 'Kylian', 'Ladji', 'Lana', 'Lassana',
            'Lenny', 'Leon', 'Loïc', 'Lou', 'Lucie', 'Maëlys', 'Malia', 'Marvin', 'Mathis', 'Mattéo',
            'Mayar', 'Mélia', 'Meriem', 'Naomie', 'Nélia', 'Olga', 'Pharell', 'Ramata', 'Ritaj', 'Ryan',
            'Sami', 'Steven', 'Suzanne', 'Théodore', 'Théophile', 'Tiago', 'Tim', 'Wassim', 'Yasmine', 'Ysée',
            'Aïden', 'Aksel', 'Albane', 'Aliya', 'Amadou', 'Anaé', 'Anastasia', 'Andrea', 'Andy', 'Anton',
            'Antonio', 'Ayoub', 'Billie', 'Bryan', 'Cécile', 'Charlie', 'Dario', 'Darius', 'Diane', 'Eden',
            'Edmond', 'Éléonore', 'Elon', 'Elsa', 'Elyssa', 'Emily', 'Emma', 'Eugénie', 'Eva', 'Evy',
            'Eyal', 'Ezra', 'Faith', 'Felix', 'Félix', 'Gael', 'Gaspard', 'Gauthier', 'Idriss', 'Issam',
            'Julien', 'Jun', 'Leandro', 'Levi', 'Leyla', 'Liza', 'Logan', 'Loris', 'Luca', 'Mady',
            'Marion', 'Marthe', 'Martin', 'Mathias', 'Merlin', 'Mira', 'Neyla', 'Nizar', 'Paloma', 'Pharell',
            'Pietro', 'Rafael', 'Rayane', 'Romane', 'Saïd', 'Sami', 'Sonia', 'Soraya', 'Swann', 'Théodore',
            'Timéo', 'Tyler', 'Ysée', 'Zakariya', 'Zayn', 'Zineb', 'Ziyad', 'Adama', 'Adel', 'Adem',
            'Agathe', 'Aïna', 'Aksel', 'Alexis', 'Alicia', 'Andrea', 'Anissa', 'Assiya', 'Aurore', 'Axel',
            'Benoît', 'Bilal', 'Bintou', 'Carl', 'Dylan', 'Élisa', 'Eloi', 'Elyne', 'Elyssa', 'Emna',
            'Enora', 'Eulalie', 'Farès', 'Giulia', 'Hamady', 'Haroun', 'Hidaya', 'Ilef', 'Isidore', 'Iyad',
            'Izaac', 'Jana', 'Jeanne', 'Jed', 'Johann', 'Jordan', 'Josué', 'Kadidja', 'Lehna', 'Léonore',
            'Leyna', 'Lisa', 'Lison', 'Loan', 'Louis', 'Lyne', 'Maëlle', 'Marcus', 'Marguerite', 'Marlon',
            'Marvin', 'Max', 'Maxine', 'May', 'Melchior', 'Meryam', 'Mira', 'Naël', 'Nava', 'Nazim',
            'Nesrine', 'Nina', 'Nora', 'Octave', 'Olivia', 'Qassim', 'Rafael', 'Ritaj', 'Ryad', 'Sandra',
            'Selim', 'Soraya', 'Teodor', 'Tina', 'Uriel', 'Valentine', 'Vladimir', 'Yaïr', 'Adem', 'Aïcha',
            'Aimé', 'Aline', 'Alya', 'Ange', 'Angelina', 'Annabelle', 'Anya', 'Assiya', 'Axel', 'Violette',
            'Benjamin', 'Yaël', 'Bérénice', 'Zéphyr', 'Bonnie', 'Abdallah', 'Céleste', 'Andrea', 'Daphné', 'Arnaud',
            'Diego', 'Ayden', 'Élia', 'Boubacar', 'Élisa', 'Cassandra', 'Emilie', 'Charly', 'Etienne', 'Darius',
            'Hafsa', 'Elia', 'Hamed', 'Emy', 'Haron', 'Eric', 'Hector', 'Fatima', 'Ilian', 'Ferdinand',
            'Ishak', 'Fleur', 'Ivan', 'Florian', 'Izïa', 'Gaïa', 'Joanne', 'Grégoire', 'Khalil', 'Hafsa',
            'Léonor', 'Hajar', 'Louane', 'Hermine', 'Loup', 'Ilhan', 'Luc', 'Ilyès', 'Maëline', 'Ismaïl',
            'Maeva', 'Jessica', 'Malo', 'Killian', 'Marion', 'Kylian', 'Mélanie', 'Lahna', 'Mikaël', 'Laurent',
            'Mila', 'Leonardo', 'Mira', 'Lily', 'Mona', 'Lise', 'Muhammed', 'Livio', 'Nahil', 'Lyam',
            'Nathanael', 'Maé', 'Nil', 'Maelys', 'Noah', 'Mahé', 'Nola', 'Malek', 'Philippa', 'Malone',
            'Rafael', 'Marine', 'Razane', 'Matisse', 'Rosalie', 'Mattéo', 'Sanaa', 'May', 'Sekou', 'Mody',
            'Seydina', 'Nabil', 'Soan', 'Naïa', 'Sofiane', 'Neïla', 'Teodor', 'Oren', 'Thiago', 'Rym',
            'William', 'Sabrina', 'Yossef', 'Siloé', 'Sacha', 'Toscane', 'Iris', 'Vladimir', 'Romy', 'Wael',
            'Octave', 'Xavier', 'Ulysse', 'Youssef', 'Hector', 'Zacharie', 'Gabin', 'Adrien', 'Pierre', 'Ahmad',
            'Lila', 'Aïda', 'Ahmed', 'Aïdan', 'Yassine', 'Alya', 'Jenna', 'Anass', 'Youssouf', 'André',
            'Augustine', 'Ange', 'Lena', 'Anthony', 'Samy', 'Bruno', 'Mélissa', 'Celia', 'Adama', 'Chahinez',
            'Esmée', 'Charlène', 'Mahault', 'Cheikh', 'Léana', 'Chiara', 'Ora', 'Christian', 'Anis', 'Cindy',
            'Matthieu', 'Clément', 'Soren', 'Cléophée', 'Yuna', 'Coumba', 'Ilyane', 'Curtis', 'Seydou', 'Dounia',
            'Aïsha', 'Éléna', 'Gloria', 'Elie', 'Nour', 'Emir', 'Ève', 'Emna', 'Mariya', 'Éric',
            'Abdou', 'Giovanni', 'Milhane', 'Guilhem', 'Samba', 'Hind', 'Andy', 'Iliana', 'Angel', 'Imrane',
            'Anouk', 'Jacob', 'Anthony', 'Jonas', 'Antoine', 'Josh', 'Aubin', 'Josué', 'Augustine', 'Juliette',
            'Barnabé', 'Justine', 'Bérénice', 'Kadiatou', 'Brayan', 'Kim', 'Carmen', 'Léa', 'Cassandre', 'Livia',
            'Charlotte', 'Lola', 'Christophe', 'Lucas', 'Clara', 'Maëlys', 'Clarisse', 'Mamou', 'Daphnée', 'Manelle',
            'Deborah', 'Marlon', 'Déborah', 'Mathieu', 'Domitille', 'Matthieu', 'El', 'Maxime', 'Elias', 'Mayar',
            'Elise', 'Meïssa', 'Ely', 'Méline', 'Enola', 'Milan', 'Flavie', 'Nahel', 'Hélène', 'Naomie',
            'Hillel', 'Neil', 'Hugues', 'Nora', 'Ibrahima', 'Othmane', 'Jules', 'Paolo', 'Julia', 'Rania',
            'Kacper', 'Reda', 'Kadiatou', 'Sasha', 'Karl', 'Sekou', 'Katia', 'Selena', 'Klara', 'Sonia',
            'Laurent', 'Thanina', 'Leo', 'Tobias', 'Lirone', 'Wesley', 'Liv', 'Yohan', 'Livia', 'Zakaria',
            'Lucas', 'Zakariya', 'Maïlys', 'Achille', 'Marina', 'Adriana', 'Michaël', 'Adriel', 'Milla', 'Alassane',
            'Naël', 'Alessia', 'Noa', 'Amélia', 'Ousmane', 'Amicie', 'Paul', 'Anaëlle', 'Philibert', 'Ange',
            'Philomène', 'Anouk', 'Quitterie', 'Anton', 'Rokia', 'Athénaïs', 'Sarah-Lou', 'Aurélien', 'Séléna', 'Carla',
            'Sofiane', 'Cléo', 'Sonia', 'Dina', 'Théa', 'Djibril', 'Théophane', 'Eléna', 'Victor', 'Elian',
            'Wandrille', 'Eloi', 'Aaron', 'Elvire', 'Aby', 'Florent', 'Adèle', 'Gaël', 'Aïssata', 'Garance',
            'Aïssatou', 'Gaspard', 'Alice', 'Harold', 'Amaury', 'Hélie', 'Amélia', 'Imad', 'Andréa', 'Ismaël',
            'Anna', 'Ivan', 'Armelle', 'Iyad', 'Augustin', 'Jayson', 'Bakary', 'Jenna', 'Bérénice', 'Joseph',
            'Caroline', 'Joyce', 'César', 'Kadidja', 'Chirine', 'Kimi', 'Clovis', 'Lehna', 'Côme', 'Lila',
            'Daouda', 'Maëlys', 'Darius', 'Maïmouna', 'Dina', 'Maïssa', 'Divine', 'Mathys', 'Djibril', 'Mayeul',
            'Dramane', 'Mayssa', 'Edgar', 'Miya', 'Zakaria', 'Ellie', 'Mory', 'Abdourahmane', 'Elodie', 'Naëlle',
            'Adrian', 'Elvire', 'Neïla', 'Adrien', 'Erin', 'Nina', 'Alexandra', 'Eya', 'Noa', 'André',
            'Gaétan', 'Noham', 'Antony', 'Hawa', 'Oskar', 'Augustine', 'Ismaël', 'Patricia', 'Camille', 'Issa',
            'Philomène', 'Celia', 'Jacob', 'Rokia', 'Yann', 'Chayma', 'Jane', 'Sana', 'Yannick', 'Constantin',
            'Jason', 'Shana', 'Adèle', 'Éléonore', 'Jonas', 'Stanislas', 'Adib', 'Elio', 'Kadiatou', 'Viktor',
            'Alfred', 'Élise', 'Kaïs', 'Xavier', 'Amine', 'Elliott', 'Kassandra', 'Yvan', 'Anna', 'Erin',
            'Kelya', 'Zayneb', 'Arié', 'Ethel', 'Kelyan', 'Aissatou', 'Augustin', 'Etienne', 'Kenzo', 'Alexandre',
            'Bastien', 'Eugénie', 'Lamia', 'Alima', 'Bianca', 'Fériel', 'Léana', 'Alon', 'Boris', 'Gustave',
            'Léonie', 'Alpha', 'Candice', 'Hasna', 'Léonore', 'Aly', 'Chris', 'Idriss', 'Lorenzo', 'Angelina',
            'Claire', 'Inaya', 'Lucie', 'Anita', 'Constance', 'Isaac', 'Manuela', 'Aurélien', 'Emma', 'Isabelle',
            'Marvin', 'Brune', 'Eugène', 'Jalil', 'Mathias', 'Carolina', 'Eve', 'Jasmine', 'Mathys', 'Célestin',
            'Ewen', 'Jenna', 'Milena', 'Christian', 'Ezio', 'Jude', 'Milla', 'Corentin', 'Fadi', 'Lancelot',
            'Nadir', 'Edouard', 'Franck', 'Laura', 'Neil', 'Édouard', 'Gabriela', 'Lévana', 'Noah', 'Edward',
            'Gautier', 'Lilia', 'Otto', 'Élio', 'Jean', 'Ludivine', 'Perrine', 'Eloi', 'Johan', 'Luna',
            'Sasha', 'Ezio', 'Juliette', 'Lya', 'Sean', 'Félix', 'Kelly', 'Marcel', 'Selma', 'Gabriela',
            'Lena', 'Marcus', 'Shaïma', 'Giovanni', 'Lévi', 'Marguerite', 'Sophia', 'Hélie', 'Leyna', 'Mariama',
            'Tara', 'Jasmine', 'Linoy', 'Marianne', 'Tiphaine', 'Jessica', 'Liv', 'Marion', 'Vladimir', 'Jonah',
            'Louison', 'Matis', 'Xavier', 'Kamil', 'Lucille', 'Mayline', 'Yara', 'Kristina', 'Maïwenn', 'Méline',
            'Yasmina', 'Kurtis', 'Marc', 'Mickaël', 'Zachary', 'Laure', 'Matias', 'Morgane', 'Abdallah', 'Layana',
            'Max', 'Mouhamed', 'Abdoul', 'Léandre', 'Maxence', 'Nadia', 'Adama', 'Leïa', 'Maxine', 'Naël',
            'Aglaé', 'Leïna', 'Meryem', 'Naïm', 'Amaury', 'Léona', 'Nada', 'Nathanael', 'Amin', 'Lisa',
            'Nael', 'Rivka', 'Aron', 'Lorenzo', 'Nawel', 'Sara', 'Badis', 'Louise', 'Nikola', 'Sarra',
            'Baptiste', 'Luc', 'Noor', 'Sean', 'Barbara', 'Lucia', 'Rafaël', 'Sixtine', 'Bertille', 'Mahamadou',
            'Raphaëlle', 'Soline', 'Bilal', 'Manil', 'Safiya', 'Soumaya', 'Célestin', 'Marianne', 'Shirel', 'Stanislas',
            'Clémentine', 'Marie', 'Simon', 'Suzie', 'Curtis', 'Matéo', 'Stella', 'Swann', 'Edouard', 'Maxime',
            'Talya', 'Thibaud', 'Élisa', 'Menahem', 'Tasnim', 'Victor', 'Eugène', 'Milan', 'Thomas', 'Viktor',
            'Héléna', 'Mohammed', 'Titouan', 'Walid', 'Issa', 'Moïse', 'Tom', 'Yani', 'Joan', 'Morgane',
            'Yanis', 'Yann', 'Kenny', 'Nawel', 'Yara', 'Youri', 'Kenzi', 'Nelson', 'Youness', 'Ysé',
            'Kenzo', 'Nélya', 'Zayd', 'Yuna', 'Kévin', 'Ortal', 'Alix', 'Zahra', 'Khalil', 'Paulin',
            'Amélia', 'Zakaria', 'Lilas', 'Pauline', 'Antoni', 'Abdoulaye', 'Lilya', 'Rafael', 'Aurélien', 'Abigaïl',
            'Line', 'Raphaëlle', 'Ayline', 'Aboubacar', 'Lisa', 'Rita', 'Benjamin', 'Aïcha', 'Livia', 'Sacha',
            'Bertille', 'Alexander', 'Louise', 'Safya', 'Capucine', 'Alicia', 'Lucile', 'Said', 'Carolina', 'Alina',
            'Maïssane', 'Salim', 'Charlotte', 'Aloys', 'Mathéo', 'Santiago', 'Cheick', 'Amina', 'Michaël', 'Sibylle',
            'Daniel', 'Amir', 'Myriam', 'Solène', 'Driss', 'Anaëlle', 'Nathan', 'Swan', 'Éléonore', 'Anaïs',
            'Nina', 'Syrine', 'Elia', 'Annabelle', 'Noémie', 'Tara', 'Elias', 'Armand', 'Octave', 'Taym',
            'Eliot', 'Aron', 'Oumar', 'Thierno', 'Elisabeth', 'Aymane', 'Quentin', 'Tyron', 'Emil', 'Blanche',
            'Rita', 'Yassin', 'Émile', 'Brune', 'Ruben', 'Aaliyah', 'Eve', 'Clémentine', 'Sam', 'Adeline',
            'Felix', 'Divine', 'Syrine', 'Adriana', 'Flora', 'Étienne', 'Tia', 'Aglaé', 'Gaspard', 'Gabriela',
            'Uma', 'Alassane', 'Ibrahima', 'Giulia', 'Wiktoria', 'André', 'Ilan', 'Gloria', 'Wissem', 'Angélina',
            'Ilyana', 'Hannah', 'Yohan', 'Assil', 'Ilyes', 'Idrissa', 'Aliénor', 'Calista', 'Jade', 'Ismaïl',
            'Alon', 'César', 'Kays', 'Joaquim', 'Anaë', 'Clara', 'Kenzo', 'Johanna', 'Audrey', 'Domitille',
            'Kiara', 'Jules', 'Ayden', 'Elio', 'Layla', 'Kaïna', 'Benoît', 'Elisa', 'Lehna', 'Kevin',
            'Blandine', 'Emilie', 'Leïna', 'Kyan', 'Célian', 'Éthan', 'Lena', 'Lazare', 'Clélia', 'Evann',
            'Léopold', 'Lenny', 'Clément', 'Ewen', 'Liel', 'Lilya', 'Coline', 'Florent', 'Lucien', 'Lior',
            'Côme', 'Gaëtan', 'Mahaut', 'Lital', 'Diana', 'Giulio', 'Mai', 'Lyne', 'Divine', 'Greta',
            'Manon', 'Matilda', 'Edward', 'Hanaé', 'Mariame', 'Matteo', 'Elie', 'Hanna', 'Mateo', 'Maxim',
            'Emil', 'Ishaq', 'Mateusz', 'Meryem', 'Emilia', 'Jade', 'Matthew', 'Neïla', 'Farès', 'Jana',
            'Mayeul', 'Nesrine', 'Ferdinand', 'Jayden', 'Melvin', 'Rana', 'Gaïa', 'Joséphine', 'Mila', 'Sacha',
            'Joey', 'Jules', 'Mona', 'Soumaya', 'John', 'Julie', 'Moustapha', 'Steven', 'Livia', 'Kenzi',
            'Ornella', 'Tristan', 'Lucas', 'Khalil', 'Owen', 'Yaya', 'Lyne', 'Kiara', 'Rachel', 'Adriano',
            'Mahaut', 'Kim', 'Rita', 'Aidan', 'Mahé', 'Lea', 'Robinson', 'Alessio', 'Manelle', 'Lehna',
            'Roxanne', 'Aliénor', 'Matthias', 'Léopold', 'Sabrina', 'Alma', 'Merlin', 'Lilian', 'Selma', 'Anaël',
            'Mika', 'Louise', 'Sira', 'Andreas', 'Mila', 'Lucile', 'Taha', 'Annabelle', 'Mohamed-Amine', 'Mael',
            'William', 'Antonia', 'Nermine', 'Maëlys', 'Youcef', 'Arya', 'Noam', 'Maeva', 'Zachary', 'Assa',
            'Noé', 'Marcel', 'Zakarya', 'Ayaan', 'Nola', 'Mariam', 'Zyad', 'Blanche', 'Ousmane', 'Massil',
            'Adama', 'Céline', 'Pia', 'Matthieu', 'Adèle', 'Charline', 'Sandro', 'Mayssane', 'Adeline', 'Cléophée',
            'Shana', 'Mehdi', 'Aïna', 'Corentin', 'Simon', 'Mia', 'Alaa', 'Daouda', 'Thomas', 'Mickael',
            'Alex', 'Dario', 'Vadim', 'Milla', 'Alexandra', 'Edmond', 'Victor', 'Nathaniel', 'Alia', 'Eleanor',
            'Yoann', 'Noa', 'Aline', 'Elliott', 'Yohann', 'Noham', 'Allya', 'Emmanuel', 'Abdou', 'Paul',
            'Ambre', 'Esther', 'Adélaïde', 'Pénélope', 'Angelo', 'Faustine', 'Alessandra', 'Rachel', 'Aron', 'Fleur',
            'Alessandro', 'Rafaël', 'Aurel', 'Guillaume', 'Alexander', 'Rivka', 'Bilel', 'Isaiah', 'Alyssa', 'Roman',
            'Binta', 'Issa', 'Amira', 'Saja', 'Boubacar', 'Iyed', 'Anael', 'Samantha', 'Carmen', 'Izia',
            'Arielle', 'Samira', 'Claire', 'Jasmine', 'Basma', 'Selma', 'Côme', 'Jean-Baptiste', 'Bertille', 'Shana',
            'Coumba', 'Kylian', 'Caroline', 'Sibylle', 'Djena', 'Lassana', 'Clovis', 'Simon', 'Élie', 'Leïla',
            'Cyrille', 'Solal', 'Esteban', 'Léna', 'Damian', 'Tess', 'Eugénie', 'Léopold', 'Darius', 'Tiago',
            'Grégoire', 'Léopoldine', 'Djibril', 'Wael', 'Haroun', 'Luc', 'Domitille', 'Yasmine', 'Héloïse', 'Lucie',
            'Eli', 'Yassine', 'Honoré', 'Mael', 'Elias', 'Zack', 'Imen', 'Marc-Antoine', 'Elina', 'Abel',
            'Isabelle', 'Marcel', 'Elsa', 'Adja', 'Ismael', 'Matias', 'Emma', 'Agnès', 'Julien', 'Maud',
            'Enora', 'Alima', 'Kassim', 'Maxence', 'Eric', 'Alpha', 'Khadija', 'Mendel', 'Erika', 'Annaëlle',
            'Laetitia', 'Mina', 'Erin', 'Anthony', 'Lauren', 'Naël', 'Esteban', 'Anton', 'Léane', 'Noah',
            'Gloria', 'Anya', 'Lison', 'Olivier', 'Halima', 'Ayana', 'Louison', 'Prune', 'Hippolyte', 'Damian',
            'Loup', 'Quitterie', 'Ian', 'Elie', 'Malone', 'Rahma', 'Ilias', 'Elisabeth', 'Mathieu', 'Rayan',
            'Ilyan', 'Elon', 'Maxine', 'Rayhana', 'Isild', 'Félicie', 'Morgane', 'Rémy', 'Jason', 'Florian',
            'Moustapha', 'Rosa', 'Karim', 'Gabrielle', 'Olive', 'Rosalie', 'Laetitia', 'Haby', 'Ora', 'Salim',
            'Lana', 'Héloïse', 'Owen', 'Samy', 'Lancelot', 'Hippolyte', 'Rachel', 'Seydou', 'Léa', 'Inayah',
            'Razane', 'Soraya', 'Leïla', 'Ismael', 'Saul', 'Syrine', 'Léonore', 'Ismail', 'Sophie', 'Théo',
            'Léopoldine', 'Ismaïl', 'Théophile', 'Thomas', 'Lirone', 'Jacob', 'Tia', 'Yara', 'Luc', 'Johan',
            'Timéo', 'Yohan', 'Ludovic', 'Judith', 'Vasco', 'Zahra', 'Lydia', 'Kassim', 'William', 'Zayd',
            'Lyes', 'Lassana', 'Wissam', 'Zineb', 'Maïwenn', 'Leonardo', 'Wissem', 'Alhassane', 'Mathurin', 'Levana',
            'Yaëlle', 'Aliénor', 'Matis', 'Lilas', 'Zakariya', 'Alina', 'Mayeul', 'Lubin', 'Abdallah', 'Amélie',
            'Miguel', 'Lucy', 'Ahmad', 'Andrea', 'Naël', 'Maël', 'Alban', 'Angèle', 'Nahil', 'Malek',
            'Alexandra', 'Axel', 'Neil', 'Malika', 'Alice', 'Badis', 'Nicole', 'Mayline', 'Alix', 'Bakary',
            'Noa', 'Mélanie', 'Aloïs', 'Baptiste', 'Nour', 'Mélia', 'Amanda', 'Ben', 'Olivier', 'Mohamed',
            'Amir', 'Calie', 'Perla', 'Nahla', 'Anatole', 'Céleste', 'Ramy', 'Ninon', 'Anne', 'Charlie',
            'Stanislas', 'Nizar', 'Antonia', 'Christopher', 'Walid', 'Pablo', 'Assiya', 'Damian', 'Yani', 'Pierre',
            'Augustin', 'Daniel', 'Yanis', 'Saïd', 'Camil', 'David', 'Yannis', 'Samuel', 'Candice', 'Djibril',
            'Yasmina', 'Sélène', 'Cécile', 'Dorian', 'Yona', 'Swann', 'César', 'Dyna', 'Yuna', 'Talia',
            'Edgar', 'Elena', 'Adil', 'Walid', 'Éléonore', 'Eléonore', 'Alessia', 'Wanda', 'Florence', 'Émile',
            'Anabelle', 'Yanis', 'Florian', 'Émilie', 'Anaé', 'Adama', 'Gabin', 'Ennio', 'Angéline', 'Adélie',
            'Gustave', 'Enzo', 'Anne', 'Ali', 'Hadrien', 'Ezra', 'Arielle', 'Alix', 'Hamidou', 'Hannah',
            'Assya', 'Anaelle', 'Idriss', 'Helena', 'Bradley', 'Antonio', 'Imene', 'Héloïse', 'Clémentine', 'Aris',
            'Ivan', 'Ines', 'Cléo', 'Astrée', 'Jack', 'Isaure', 'Élie', 'Aurel', 'Jonas', 'Isra',
            'Elise', 'Basile', 'Jonathan', 'Israa', 'Emilie', 'Brune', 'Jude', 'James', 'Ethel', 'Célestine',
            'Julian', 'Kenzi', 'Feryel', 'Cheikh', 'Kyle', 'Koumba', 'George', 'Cléo', 'Lahna', 'Leonardo',
            'Halima', 'Damian', 'Lara', 'Léo-Paul', 'Ismael', 'Domitille', 'Laurine', 'Lia', 'Ivan', 'Eden',
            'Lilya', 'Louisa', 'Jacqueline', 'Eliakim', 'Linoy', 'Louison', 'Jérémy', 'Elise', 'Manel', 'Lucien',
            'Johann', 'Emmanuelle', 'Mathys', 'Maia', 'Joséphine', 'Eric', 'Matilda', 'Maxence', 'Katia', 'Fanny',
            'Mia', 'Méline', 'Kelly', 'Félicité', 'Michaël', 'Milo', 'Keyla', 'Filip', 'Mohamed-Ali', 'Mira',
            'Léon', 'Gaëtan', 'Mouhamadou', 'Miya', 'Lise', 'Garance', 'Moustapha', 'Mohamed-Lamine', 'Loïs', 'Gary',
            'Nicolas', 'Morgan', 'Lou', 'George', 'Nikita', 'Naïl', 'Louca', 'Haroun', 'Nil', 'Nathaniel',
            'Lucille', 'Haya', 'Noham', 'Noémie', 'Maëlie', 'Hippolyte', 'Perle', 'Paolo', 'Maëlle', 'Isabelle',
            'Quitterie', 'Raphaël', 'Maéva', 'Ishaq', 'Romain', 'Sandra', 'Maïssane', 'Ivy', 'Ruben', 'Sara',
            'Maïwenn', 'James', 'Salma', 'Soan', 'Marcel', 'Jean', 'Shana', 'Sophie', 'Maria', 'Jérémy',
            'Souleymane', 'Soren', 'Marie', 'Joey', 'Thalia', 'Tessa', 'Mathys', 'Johann', 'Théodora', 'Zakaria',
            'Mélodie', 'Kadidja', 'Thibault', 'Adama', 'Merlin', 'Lahna', 'Violette', 'Adèle', 'Mouhamed', 'Laïa',
            'Yohan', 'Akram', 'Nada', 'Lana', 'Youcef', 'Ali', 'Naïma', 'Léandre', 'Zeynab', 'Aline',
            'Nicolas', 'Lenny', 'Abdoul', 'Aly', 'Noé', 'Lila', 'Adélie', 'Amelia', 'Priscille', 'Liv',
            'Adil', 'Amina', 'Rémi', 'Livia', 'Adriano', 'Anaya', 'Saïd', 'Lucia', 'Aïda', 'Arthus',
            'Sharon', 'Malek', 'Aimée', 'Assia', 'Stéphane', 'Marcus', 'Aliénor', 'Ayline', 'Tesnime', 'Marguerite',
            'Alyssa', 'Bonnie', 'Théa', 'Marwan', 'Célestin', 'Bosco', 'Thibault', 'Mateo', 'Charlize', 'Charline',
            'Tom', 'Melvil', 'Chris', 'Chloé', 'Ahmed', 'Mickaël', 'Daniel', 'Clémence', 'Alexander', 'Morgan',
            'Daphnée', 'Daouda', 'Alexia', 'Nélya', 'Eden', 'David', 'Alfred', 'Nine', 'Edouard', 'Dylan',
            'Alicia', 'Noa', 'Élisa', 'Elena', 'Alissa', 'Noémie', 'Elizabeth', 'Eli', 'Alvin', 'Paolo',
            'Eloïse', 'Ellie', 'Alya', 'Prince', 'Elyes', 'Elyan', 'Amin', 'Rayane', 'Emna', 'Enora',
            'Anna', 'Rosa', 'Erwan', 'Fanny', 'Athéna', 'Ryan', 'Fatou', 'Fatma', 'Aurélie', 'Selena',
            'Faustine', 'Félicie', 'Benjamin', 'Sibylle', 'Flore', 'Gabriel', 'Brandon', 'Solal', 'Garance', 'Georgia',
            'Capucine', 'Syrine', 'Hedi', 'Grace', 'Chanel', 'Zayd', 'Hugo', 'Ibrahima', 'Chelsea', 'Louis',
            'Ilyane', 'Isra', 'Cléo', 'Oscar', 'Isaure', 'Joël', 'Dan', 'Juliette', 'Jaden', 'Joyce',
            'Dimitri', 'Côme', 'Janelle', 'Kamil', 'Elia', 'Ethan', 'Jannah', 'Kenza', 'Eline', 'Aya',
            'Jason', 'Laetitia', 'Élise', 'Brune', 'Joana', 'Layane', 'Elsa', 'Liv', 'Joy', 'Lior',
            'Emy', 'Henri', 'Kevin', 'Liora', 'Evelyne', 'Noa', 'Lena', 'Loane', 'Félix', 'Alissa',
            'Sara', 'Léo', 'Makan', 'Gauthier', 'Alya', 'Pauline', 'Liana', 'Mark', 'Gianni', 'Amir',
            'Hortense', 'Lital', 'Marvin', 'Halima', 'Arié', 'Milan', 'Lysandre', 'Marwa', 'Sasha', 'Ben',
            'Aurèle', 'Maelys', 'Mathys', 'Shana', 'Cindy', 'Mehdi', 'Malo', 'Maud', 'Soraya', 'Clarence',
            'Mahaut', 'Marc', 'Maxime', 'Syrine', 'Dana', 'Mya', 'Matéo', 'Meyron', 'Théa', 'Diana',
            'Ilan', 'Melchior', 'Miya', 'Yona', 'Elena', 'Janna', 'Myriam', 'Noam', 'Zachary', 'Elisa',
            'Jonathan', 'Pedro', 'Océane', 'Abdoulaye', 'Elora', 'Matéo', 'Rayan', 'Romie', 'Aïnhoa', 'Emily',
            'Issam', 'Ritej', 'Sadio', 'Albane', 'Eulalie', 'Line', 'Roch', 'Souleymane', 'Amadou', 'Harry',
            'Solène', 'Sabri', 'Swan', 'Aminata', 'Hatem', 'Eli', 'Safiya', 'Victoria', 'Angèle', 'Hector',
            'Erwan', 'Sandra', 'Wendy', 'Anis', 'Ismail', 'Maxime', 'Selena', 'Yani', 'Anne-Laure', 'Julien',
            'Abdallah', 'Sherine', 'Yassin', 'Anton', 'Kayna', 'Célestin', 'Sienna', 'Zohra', 'Antony', 'Lilou',
            'Flore', 'Tessa', 'Abdel', 'Apolline', 'Loris', 'Tasnim', 'Théa', 'Abdoulaye', 'Ari', 'Mahault',
            'Andreas', 'Yuna', 'Abigaëlle', 'Asma', 'Manel', 'Lassana', 'Yvan', 'Adèle', 'Ben', 'Marguerite',
            'Léopoldine', 'Abigaïl', 'Adeline', 'Benjamin', 'Marwane', 'Édouard', 'Aïda', 'Anaïs', 'Bruno', 'Matéo',
            'Emil', 'Albane', 'Andréa', 'Calixte', 'Mendel', 'Hassan', 'Alphonse', 'Angelina', 'Christian', 'Nadine',
            'Mady', 'Amelia', 'Auguste', 'Christophe', 'Naïl', 'Alexi', 'André', 'Aurèle', 'Cindy', 'Nathaël',
            'Aurèle', 'Angèle', 'Charlie', 'Clothilde', 'Noémie', 'Avi', 'Anna', 'Cloé', 'Edgar', 'Octave',
            'Aymar', 'Anne', 'Dalia', 'Etienne', 'Pablo', 'Ayrton', 'Anouck', 'Damien', 'Fatou', 'Penda',
            'Bertille', 'Assia', 'Dan', 'Faustine', 'Perrine', 'Betty', 'Aude', 'Dorian', 'Florence', 'Pierre',
            'Brice', 'Carmen', 'Eliakim', 'Geoffroy', 'Rami', 'Camélia', 'Céline', 'Elijah', 'Hugues', 'Rébecca',
            'Carlotta', 'Dania', 'Emma', 'Iliana', 'Robin', 'Colin', 'Éléonore', 'Étienne', 'Imen', 'Ryad',
            'Curtis', 'Eliana', 'Gabriel', 'Inés', 'Samy', 'Diane', 'Eliza', 'Gaëlle', 'Karim', 'Scarlett',
            'Eliette', 'Emilie', 'Hillel', 'Kenza', 'Selma', 'Erwan', 'Emilio', 'Idriss', 'Lauryn', 'Sidy',
            'Ethan', 'Frida', 'Ilyana', 'Léon', 'Swann', 'Félicité', 'Gaspard', 'Jessim', 'Mathilda', 'Syrine',
            'Flavio', 'George', 'Katia', 'Matisse', 'Tara', 'Gabrielle', 'Jean', 'Lana', 'Matthew', 'Thaïs',
            'Germain', 'John', 'Layla', 'Maxine', 'Yanni', 'Goundo', 'Joseph', 'Léa', 'Maya', 'Zachary',
            'Gwendal', 'Kenzo', 'Leandro', 'Mouna', 'Zahra', 'Hector', 'Layane', 'Leila', 'Paola', 'Alfred',
            'Sara', 'Lilian', 'Lévi', 'Philippe', 'Alyssa', 'Siham', 'Louison', 'Liana', 'Rania', 'Antoine',
        ]
