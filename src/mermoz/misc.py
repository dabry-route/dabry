import sys
import numpy as np
from math import pi, acos, cos, sin, floor

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


TRAJ_PMP = 'pmp'
TRAJ_INT = 'integral'

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


def geodesic_distance(*args, mode='rad'):
    """
    Computes the great circle distance between two points on the earth surface
    Arguments may be given in (lon1, lat1, lon2, lat2)
    or in (ndarray(lon1, lat1), ndarray(lon2, lat2))
    :param mode: Whether lon/lat given in degrees or radians
    :return: Great circle distance in meters
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
        print('Incorrect argument format', file=sys.stderr)
        exit(1)
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
    res = acos(sin(lambda1) * sin(lambda2) + cos(lambda1) * cos(lambda2) * cos(delta_phi)) * EARTH_RADIUS
    return res


def distance(x1, x2, coords=COORD_CARTESIAN):
    if coords == COORD_GCS:
        # x1, x2 shall be vectors (lon, lat) in radians
        return geodesic_distance(x1, x2)
    else:
        # x1, x2 shall be cartesian vectors in meters
        return np.linalg.norm(x1 - x2)


def decorate(ax, title=None, xlab=None, ylab=None, legend=None, xlim=None, ylim=None, min_yspan=None):
    ax.xaxis.grid(color='k', linestyle='-', linewidth=0.2)
    ax.yaxis.grid(color='k', linestyle='-', linewidth=0.2)
    if xlab: ax.xaxis.set_label_text(xlab)
    if ylab: ax.yaxis.set_label_text(ylab)
    if title: ax.set_title(title, {'fontsize': 8})
    if legend != None: ax.legend(legend, loc='best')
    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if min_yspan != None: ensure_yspan(ax, min_yspan)


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
