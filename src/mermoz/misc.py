import sys
import time

import numpy as np
from math import pi, acos, cos, sin, floor, atan2

from numpy import ndarray

COORD_CARTESIAN = 'cartesian'
COORD_GCS = 'gcs'
COORDS = [COORD_CARTESIAN, COORD_GCS]

U_METERS = 'meters'
U_DEG = 'degrees'
U_RAD = 'rad'
UNITS = [U_METERS, U_DEG, U_RAD]
DEG_TO_RAD = pi / 180.
RAD_TO_DEG = 180. / pi
AIRSPEED_DEFAULT = 23.  # [m/s]


def to_0_360(angle):
    """
    Bring angle to 0-360 domain
    :param angle: Angle in degrees
    :return: Same angle in 0-360
    """
    return angle - 360. * floor(angle / 360.)


def rectify(a, b):
    """
    Bring angles to 0-720 and keep order
    :param a: First angle in degrees
    :param b: Second angle in degrees
    :return: (a, b) but between 0 and 720 degrees
    """
    aa = to_0_360(a)
    bb = to_0_360(b)
    if aa > bb:
        bb += 360.
    return aa, bb


def ang_principal(a):
    """
    Bring angle to ]-pi,pi]
    :param a: Angle in radians
    :return: Equivalent angle in the target interval
    """
    return np.angle(np.cos(a) + 1j * np.sin(a))


def angular_diff(a1, a2):
    """
    The measure of real angular difference between a1 and a2
    :param a1: First angle in radians
    :param a2: Second angle in radians
    :return: Angular difference
    """
    b1 = ang_principal(a1)
    b2 = ang_principal(a2)
    if abs(b2 - b1) <= np.pi:
        return abs(b2 - b1)
    else:
        return 2 * np.pi - abs(b2 - b1)


def linspace_sph(a1, a2, N):
    """
    Return a ndarray consisting of a linspace of angles between the two angles
    but distributed on the geodesic between angles on the circle (shortest arc)
    :param a1: First angle in radians
    :param a2: Second angle in radians
    :param N: Number of discretization points
    :return: ndarray of angles in radians
    """
    b1 = ang_principal(a1)
    b2 = ang_principal(a2)
    b_min = min(b1, b2)
    delta_b = angular_diff(b1, b2)
    if abs(b2 - b1) <= np.pi:
        return np.linspace(b_min, b_min + delta_b, N)
    else:
        return np.linspace(b_min - delta_b, b_min, N)


TRAJ_PMP = 'pmp'
TRAJ_INT = 'integral'
TRAJ_PATH = 'path'
TRAJ_OPTIMAL = 'optimal'

EARTH_RADIUS = 6378.137e3  # [m] Earth equatorial radius


def ensure_coords(coords):
    if coords not in COORDS:
        print(f'Unknown coords type "{coords}"')
        exit(1)


def ensure_units(units):
    if units not in UNITS:
        print(f'Unknown units "{units}"')
        exit(1)


def ensure_compatible(coords, units):
    if coords == COORD_CARTESIAN:
        if units not in [U_METERS]:
            print(f'Uncompatible coords "{coords}" and grid units "{units}"')
            exit(1)
    elif coords == COORD_GCS:
        if units not in [U_RAD, U_DEG]:
            print(f'Uncompatible coords "{coords}" and grid units "{units}"')
            exit(1)


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

    factor = DEG_TO_RAD if mode == 'deg' else 1.
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
        return pi * EARTH_RADIUS
    return arg


def geodesic_distance(*args, mode='rad'):
    """
    Computes the great circle distance between two points on the earth surface
    :param args: May be given in (lon1, lat1, lon2, lat2)
    or in (ndarray(lon1, lat1), ndarray(lon2, lat2))
    :param mode: Whether lon/lat given in degrees or radians
    :return: Great circle distance in meters
    """
    c_angle = central_angle(args, mode=mode)
    return acos(c_angle) * EARTH_RADIUS


def proj_ortho(lon, lat, lon0, lat0):
    """
    :param lon: Longitude in radians
    :param lat: Latitude in radians
    :param lon0: Longitude of center in radians
    :param lat0: Latitude of center in radians
    :return: Projection in meters
    """
    return EARTH_RADIUS * np.array((np.cos(lat) * np.sin(lon - lon0),
                                    np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)))


def proj_ortho_inv(x, y, lon0, lat0):
    rho = np.sqrt(x ** 2 + y ** 2)
    c = np.arcsin(rho / EARTH_RADIUS)
    return np.array((lon0 + np.arctan(
        x * np.sin(c) / (rho * np.cos(c) * np.cos(lat0) - y * np.sin(c) * np.sin(lat0))),
                     np.arcsin(np.cos(c) * np.sin(lat0) + y * np.sin(c) * np.cos(lat0) / rho)))


def d_proj_ortho(lon, lat, lon0, lat0):
    return EARTH_RADIUS * np.array(
        ((np.cos(lat) * np.cos(lon - lon0), -np.sin(lat) * np.sin(lon - lon0)),
         (np.sin(lat0) * np.cos(lat) * sin(lon - lon0),
          np.cos(lat0) * np.cos(lat) + np.sin(lat0) * np.sin(lat) * np.cos(lon - lon0))))


def d_proj_ortho_inv(x, y, lon0, lat0):
    lon, lat = tuple(proj_ortho_inv(x, y, lon0, lat0))
    det = EARTH_RADIUS ** 2 * (np.cos(lat0) * np.cos(lat) ** 2 * np.cos(lon - lon0) +
                               np.sin(lat0) * np.sin(lat) * np.cos(lat))
    dp = d_proj_ortho(lon, lat, lon0, lat0)
    return 1 / det * np.array(((dp[1, 1], -dp[0, 1]), (-dp[1, 0], dp[0, 0])))


def middle(x1, x2, coords):
    if coords == COORD_CARTESIAN:
        # x1, x2 shall be cartesian vectors in meters
        return 0.5 * (x1 + x2)
    else:
        # Assuming coords == COORD_GCS
        # x1, x2 shall be vectors (lon, lat) in radians
        bx = np.cos(x2[1]) * np.cos(x2[0] - x1[0])
        by = np.cos(x2[1]) * np.sin(x2[0] - x1[0])
        return x1[0] + atan2(by, np.cos(x1[1]) + bx), \
               atan2(np.sin(x1[1]) + np.sin(x2[1]), np.sqrt((np.cos(x1[1]) + bx) ** 2 + by ** 2))


def distance(x1, x2, coords):
    if coords == COORD_GCS:
        # x1, x2 shall be vectors (lon, lat) in radians
        return geodesic_distance(x1, x2)
    else:
        # Assuming coords == COORD_CARTESIAN
        # x1, x2 shall be cartesian vectors in meters
        return np.linalg.norm(x1 - x2)


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
    if min_yspan is not None: ensure_yspan(ax, min_yspan)


def ensure_yspan(ax, yspan):
    ymin, ymax = ax.get_ylim()
    if ymax - ymin < yspan:
        ym = (ymin + ymax) / 2
        ax.set_ylim(ym - yspan / 2, ym + yspan / 2)


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


def heading_opti(x, p, t, coords):
    if coords == COORD_CARTESIAN:
        v = -p / np.linalg.norm(p)
        res = np.arctan2(v[1], v[0])
    else:  # coords == COORD_GCS
        v = - np.diag([1 / cos(x[1]), 1.]) @ p
        v = v / np.linalg.norm(v)
        res = np.pi / 2. - np.arctan2(v[1], v[0])
    return res


def airspeed_opti(p, cost='dobrokhodov'):
    if cost == 'dobrokhodov':
        pn = np.linalg.norm(p)
        return airspeed_opti_(pn)
    elif cost == 'subramani':
        pn = np.linalg.norm(p)
        return pn / 2.


def airspeed_opti_(pn):
    kp1 = 0.05
    kp2 = 1000
    return np.sqrt(pn / (6 * kp1) + np.sqrt(kp2 / (3 * kp1) + pn ** 2 / (36 * kp1 ** 2)))


def power(airspeed, cost='dobrokhodov'):
    if cost == 'dobrokhodov':
        kp1 = 0.05
        kp2 = 1000
        return kp1 * airspeed ** 3 + kp2 / airspeed
    elif cost == 'subramani':
        return airspeed ** 2


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
    from mermoz.trajectory import Trajectory
    return Trajectory(np.zeros(points.shape[0]),
                      points,
                      np.zeros(points.shape[0]),
                      points.shape[0] - 1,
                      coords=COORD_CARTESIAN,
                      type='optimal')


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def collision(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


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
        t_ab = (x - a[0]) / (b[0] - a[0])
    else:
        t_ab = (y - a[1]) / (b[1] - a[1])
    if d[1] - c[1] < 1e-5 * ref_dim:
        t_cd = (x - c[0]) / (d[0] - c[0])
    else:
        t_cd = (y - c[1]) / (d[1] - c[1])
    return (x, y), t_ab, t_cd


def time_fmt(duration):
    """
    Formats duration to proper units
    :param duration: Duration in seconds
    :return: String representation of duration in proper units
    """
    if duration < 200:
        return f'{duration:.2f}s'
    elif duration < 60 * 200:
        return f'{duration / 60:.2f}min'
    else:
        return f'{duration / 3600:.2f}h'


class Chrono:
    def __init__(self, no_verbose=False):
        self.t_start = 0.
        self.t_end = 0.
        self.verbose = not no_verbose

    def start(self, msg=''):
        if self.verbose:
            print(f'[*] {msg}')
        self.t_start = time.time()

    def stop(self):
        self.t_end = time.time()
        diff = self.t_end - self.t_start
        if self.verbose:
            print(f'[*] Done ({diff:.3f}s)')
        return diff
