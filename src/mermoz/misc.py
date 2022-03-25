from math import pi

COORD_CARTESIAN = 'cartesian'
COORD_GCS = 'gcs'
COORDS = [COORD_CARTESIAN, COORD_GCS]

U_METERS = 'meters'
U_DEG = 'degrees'
U_RAD = 'rad'
UNITS = [U_METERS, U_DEG, U_RAD]
DEG_TO_RAD = pi / 180.
RAD_TO_DEG = 180. / pi

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
