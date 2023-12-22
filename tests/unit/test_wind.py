import unittest
import numpy as np
from dabry.flowfield import DiscreteFF
from dabry.misc import Utils


class TestFF(unittest.TestCase):

    def test_create(self):
        nt, nx, ny = 10, 20, 20
        np.random.seed(42)
        values = np.random.random((nt, nx, ny, 2))
        bounds = np.random.random((3, 2))
        coords = Utils.COORD_CARTESIAN
        wind = DiscreteFF(values, bounds, coords)


if __name__ == '__main__':
    unittest.main()
