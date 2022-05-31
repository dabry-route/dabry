import sys
from math import pi, acos, cos, sin, floor

COORD_CARTESIAN = 'cartesian'
COORD_GCS = 'gcs'
COORDS = [COORD_CARTESIAN, COORD_GCS]

U_METERS = 'meters'
U_DEG = 'degrees'
U_RAD = 'rad'
UNITS = [U_METERS, U_DEG, U_RAD]
DEG_TO_RAD = pi / 180.
RAD_TO_DEG = 180. / pi


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
    delta_phi -= int(delta_phi/pi)*pi
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
