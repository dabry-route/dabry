from abc import ABC
import numpy as np
from mermoz.misc import *


class Obstacle(ABC):

    def __init__(self, value_func, ref_point, d_value_func=None, l_ref=None):
        self.value = value_func
        # Reference point within obstacle to compute cycle around the obstacle
        self.ref_point = np.zeros(ref_point.shape)
        self.ref_point[:] = ref_point
        # Analytical gradient
        self._d_value = d_value_func
        # Reference length for finite differencing
        self.l_ref = 1 if l_ref is None else l_ref
        self.eps = self.l_ref * 1e-5

    def value(self, x):
        """
        Return a negative value if within obstacle, positive if outside and null at the border
        :param x: Position at which to get value (1D numpy array)
        :return: Obstacle function value
        """
        pass

    def d_value(self, x):
        """
        Derivative of obstacle value function
        :param x: Position at which to get derivative (1D numpy array)
        :return: Gradient of obstacle function at point
        """
        if self._d_value is not None:
            return self._d_value(x)
        else:
            # Finite differencing, centered scheme
            n = x.shape[0]
            emat = np.diag(n * (self.eps,))
            grad = np.zeros(n)
            for i in range(n):
                grad[i] = (self.value(x + emat[i]) - self.value(x - emat[i])) / (2 * self.eps)
            return grad

    def update_lref(self, l_ref):
        self.l_ref = l_ref
        self.eps = l_ref * 1e-5


class CircleObs(Obstacle):
    """
    Circle obstacle defined by center and radius
    """

    def __init__(self, center, radius):
        self.center = np.zeros(center.shape)
        self.center[:] = center
        self.radius = radius
        self._sqradius = radius ** 2
        super().__init__(self.value, self.center, self.d_value)

    def value(self, x):
        return (x[0] - self.center[0]) ** 2 + (x[1] - self.center[1]) ** 2 - self._sqradius

    def d_value(self, x):
        return x - self.center


class FrameObs(Obstacle):
    """
    Rectangle obstacle acting as a frame
    """

    def __init__(self, bl, tr):
        self.bl = np.zeros(bl.shape)
        self.bl[:] = bl
        self.tr = np.zeros(tr.shape)
        self.tr[:] = tr
        self.center = 0.5 * (bl + tr)
        self.factor = np.diag((1 / (self.tr[0] - self.bl[0]), 1 / (self.tr[1] - self.bl[1])))
        super().__init__(self.value, self.center, self.d_value)

    def value(self, x):
        return min(x[0] - self.bl[0], self.tr[0] - x[0], x[1] - self.bl[1], self.tr[1] - x[1])

    def d_value(self, x):
        xx = self.factor @ (x - self.center)
        a, b = xx[0], xx[1]
        # Going clockwise through cases
        if a > b and a > -b:
            return np.array((1., 0.))
        elif b < a < -b:
            return np.array((0., -1.))
        elif a < b and a < -b:
            return np.array((-1., 0.))
        else:
            return np.array((0., 1.))


class GreatCircleObs(Obstacle):

    def __init__(self, p1, p2):
        X1 = np.array((EARTH_RADIUS * np.cos(p1[0]) * np.cos(p1[1]),
                       EARTH_RADIUS * np.sin(p1[0]) * np.cos(p1[1]),
                       EARTH_RADIUS * np.sin(p1[1])))
        X2 = np.array((EARTH_RADIUS * np.cos(p2[0]) * np.cos(p2[1]),
                       EARTH_RADIUS * np.sin(p2[0]) * np.cos(p2[1]),
                       EARTH_RADIUS * np.sin(p2[1])))
        self.dir_vect = -np.cross(X1, X2)
        self.dir_vect /= np.linalg.norm(self.dir_vect)
        super().__init__(self.value, np.zeros(2), self.d_value)

    def value(self, x):
        X = np.array((EARTH_RADIUS * np.cos(x[0]) * np.cos(x[1]),
                      EARTH_RADIUS * np.sin(x[0]) * np.cos(x[1]),
                      EARTH_RADIUS * np.sin(x[1])))
        return X @ self.dir_vect

    def d_value(self, x):
        d_dphi = np.array((-EARTH_RADIUS * np.sin(x[0]) * np.cos(x[1]),
                           EARTH_RADIUS * np.cos(x[0]) * np.cos(x[1]),
                           0))
        d_dlam = np.array((-EARTH_RADIUS * np.cos(x[0]) * np.sin(x[1]),
                           -EARTH_RADIUS * np.sin(x[0]) * np.sin(x[1]),
                           EARTH_RADIUS * np.cos(x[1])))
        return np.array((self.dir_vect @ d_dphi, self.dir_vect @ d_dlam))


class ParallelObs(Obstacle):

    def __init__(self, lat, up):
        """
        :param lat: Latitude in radians
        :param up: True if accessible domain is above latitude, False else
        """
        self.lat = lat
        self.up = up
        super().__init__(self.value, np.zeros(2), self.d_value)

    def value(self, x):
        if self.up:
            return x[1] - self.lat
        else:
            return self.lat - x[1]

    def d_value(self, x):
        if self.up:
            return np.array((0, 1))
        else:
            return np.array((0, -1))


class MeridianObs(Obstacle):

    def __init__(self, lon, right):
        """
        :param lon: Longitude in radians
        :param right: True if accessible domain is the half sphere in Earth's rotation direction from given longitude,
        False if it is the contrary (i.e. lon = 0, right = True then Paris is accessible but New York is not)
        """
        self.lon = lon
        self.right = right
        self.z = np.cos(lon) + 1j * np.sin(lon)
        super().__init__(self.value, np.zeros(2), self.d_value)

    def value(self, x):
        zx = np.cos(x[0]) + 1j * np.sin(x[0])
        # Cross product
        cp = (self.z.conjugate() * zx).imag
        return cp if self.right else -cp

    def d_value(self, x):
        zx = np.cos(x[0]) + 1j * np.sin(x[0])
        # Dot product
        dp = (self.z * zx.conjugate()).real
        return np.array((dp, 0)) if self.right else -np.array((dp, 0))


class MaxiObs(Obstacle):

    def __init__(self, l_obs):
        """
        Obstacle defined as intersection of obstacles
        :param l_obs: list of Obstacles
        """
        self.l_obs = l_obs
        ref_point = 1 / len(self.l_obs) * sum([obs.ref_point for obs in self.l_obs])
        super().__init__(self.value, ref_point, d_value_func=self.d_value)

    def value(self, x):
        return max([obs.value(x) for obs in self.l_obs])

    def d_value(self, x):
        i_max = max(range(len(self.l_obs)), key=lambda i: self.l_obs[i].value(x))
        return self.l_obs[i_max].d_value(x)


class LSEMaxiObs(Obstacle):

    def __init__(self, l_obs):
        """
        Obstacle defined as Log Sum Exp of obstacles
        :param l_obs: list of Obstacles
        """
        self.l_obs = l_obs
        ref_point = 1 / len(self.l_obs) * sum([obs.ref_point for obs in self.l_obs])
        self.factor = 1e5
        super().__init__(self.value, ref_point, d_value_func=self.d_value)

    def value(self, x):
        return self.factor * np.log(sum([np.exp(obs.value(x) / self.factor) for obs in self.l_obs]))

    def d_value(self, x):
        s = sum([np.exp(obs.value(x) / self.factor) for obs in self.l_obs])
        weights = [np.exp(obs.value(x) / self.factor) / s for obs in self.l_obs]
        return self.factor * sum([obs.d_value(x) * weights[i] for i, obs in enumerate(self.l_obs)])


class MeanObs(Obstacle):

    def __init__(self, l_obs):
        """
        Obstacle defined as mean of obstacles
        :param l_obs: list of Obstacles
        """
        self.l_obs = l_obs
        ref_point = 1 / len(self.l_obs) * sum([obs.ref_point for obs in self.l_obs])
        super().__init__(self.value, ref_point, d_value_func=self.d_value)

    def value(self, x):
        return 1 / len(self.l_obs) * sum([obs.value(x) for obs in self.l_obs])

    def d_value(self, x):
        return 1 / len(self.l_obs) * sum([obs.d_value(x) for obs in self.l_obs])
