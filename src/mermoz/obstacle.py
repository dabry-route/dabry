from abc import ABC
import numpy as np


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
        super().__init__(self.value, self.center, self.d_value)

    def value(self, x):
        return min(x[0] - self.bl[0], self.tr[0] - x[0], x[1] - self.bl[1], self.tr[1] - x[1])

    def d_value(self, x):
        xx = x - self.center
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
