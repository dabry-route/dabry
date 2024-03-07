import numpy as np
from dabry.flowfield import DiscreteFF
from dabry.misc import Coords


def test_create():
    nt, nx, ny = 10, 20, 20
    np.random.seed(42)
    values = np.random.random((nt, nx, ny, 2))
    bounds = np.random.random((3, 2))
    coords = Coords.CARTESIAN
    DiscreteFF(values, bounds, coords)
