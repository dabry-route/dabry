from abc import ABC, abstractmethod
import random
from math import atan2

from mermoz.wind import Wind, UniformWind
from mermoz.misc import *


class Feedback(ABC):
    """
    Defines feedback laws for given models
    """

    def __init__(self,
                 dimension: int,
                 wind: Wind):
        """
        :param dimension: The dimension of the control
        :param wind: The wind conditions
        """
        self.dim = dimension
        self.wind = wind

    @abstractmethod
    def value(self, t: float, x: ndarray) -> ndarray:
        """
        :param t: The time at which to compute the control
        :param x: The state
        :return: The feedback
        """
        pass


class AirspeedLaw(ABC):
    """
    Defines an airspeed law
    """

    @abstractmethod
    def value(self, t: float, x: ndarray) -> float:
        """
        :param t: The time at which to compute the airspeed
        :param x: The state
        :return: The corresponding airspeed
        """
        pass


class ZermeloPMPFB(Feedback):

    def __init__(self,
                 wind: Wind,
                 v_a: float,
                 p_y: float):
        """
        :param v_a: UAV airspeed in m/s
        :param p_y: Feedback parameter (from PMP)
        """
        super().__init__(1, wind)
        self.v_a = v_a
        self.p_y = p_y

    def value(self, t, x):
        wv = self.wind.value(x)[1]
        sin_value = -self.p_y * self.v_a / (1 + self.p_y * wv)
        if sin_value > 1.:
            raise ValueError("ZermeloPMPFB feedback value : sin value above 1")
        elif sin_value < -1.:
            raise ValueError("ZermeloPMPFB feedback value : sin value below -1")
        return np.arcsin(sin_value)


class ConstantFB(Feedback):

    def __init__(self, value):
        super().__init__(1, Wind())
        self._value = value

    def value(self, t, x):
        return self._value


class RandomFB(Feedback):

    def __init__(self, lower, upper, seed=42):
        self.lower = lower
        self.upper = upper
        random.seed(seed)
        super().__init__(1, Wind())

    def value(self, t, x):
        return random.uniform(self.lower, self.upper)


class FixedHeadingFB(Feedback):
    """
    Defines a control law to follow a straight line from initial position
    defined by its angle to the x-axis.
    When the wind value is too high for the UAV to keep its line, the
    control law steers perpendicular to the target direction
    """

    def __init__(self, wind, v_a: float, initial_steering: float, coords: str):
        """
        :param v_a: The UAV airspeed in m/s
        :param initial_steering: The heading to keep constant in rad
        """
        super().__init__(1, wind)
        self.v_a = v_a
        self.theta_0 = initial_steering
        self.coords = coords

    def value(self, t, x):
        if self.coords == COORD_CARTESIAN:
            e_theta_0 = np.array([np.cos(self.theta_0), np.sin(self.theta_0)])
        else:
            # coords gcs
            e_theta_0 = np.array([np.sin(self.theta_0), np.cos(self.theta_0)])
        wind = self.wind.value(t, x)
        wind_ortho = np.cross(e_theta_0, wind)
        r = -wind_ortho / self.v_a
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = np.arcsin(r)
            if self.coords == COORD_GCS:
                res *= -1
        res += self.theta_0
        return res


class GreatCircleFB(Feedback):
    """
    Control law for GCS problems only.
    Tries to stay on a great circle when wind allows it.
    """

    def __init__(self, wind, v_a, target):
        super().__init__(1, wind)
        self.v_a = v_a
        self.lon_t, self.lat_t = target

    def value(self, t, x: ndarray) -> ndarray:
        # First get the desired heading to follow great circle
        lon, lat = x
        lon_t, lat_t = self.lon_t, self.lat_t
        u0 = np.arctan(1. / (np.cos(lat) * np.tan(lat_t) / np.sin(lon_t - lon) - np.sin(lat) / np.tan(lon_t - lon)))

        e_0 = np.array([np.sin(u0), np.cos(u0)])
        wind = self.wind.value(t, x)
        wind_ortho = np.cross(e_0, wind)
        r = -wind_ortho / self.v_a
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = -np.arcsin(r)
        res += u0
        return res


class GSTargetFB(Feedback):
    """
    Control law trying to put ground speed vector towards a fixed target
    """

    def __init__(self, wind, v_a: float, target: ndarray, coords: str):
        super().__init__(1, wind)
        self.v_a = v_a
        self.target = np.zeros(2)
        self.target[:] = target
        self.coords = coords
        self.zero_ceil = 1e-3

    def value(self, t, x: ndarray):
        # Assuming GCS
        if self.coords == COORD_GCS:
            # Got to 3D cartesian assuming spherical earth
            lon, lat = x[0], x[1]
            X3 = EARTH_RADIUS * np.array((cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)))
            # Vector normal to earth at position
            e_phi = np.array((-sin(lon), cos(lon), 0.))
            e_lambda = np.array((-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)))
            lon, lat = self.target[0], self.target[1]
            X_target3 = EARTH_RADIUS * np.array((cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)))
            e_target = np.zeros(2)
            e_target[0] = (X_target3 - X3) @ e_phi
            e_target[1] = (X_target3 - X3) @ e_lambda
        else:
            #  self.coords == COORD_CARTESIAN
            e_target = np.zeros(2)
            e_target[:] = self.target - x

        if np.linalg.norm(e_target) < self.zero_ceil:
            return 0.

        e_target = e_target / np.linalg.norm(e_target)
        wind = self.wind.value(t, x)
        wind_ortho = np.cross(e_target, wind)
        r = -wind_ortho / self.v_a
        if r > 1.:
            res = np.pi / 2.
        elif r < -1.:
            res = -np.pi / 2.
        else:
            res = np.arcsin(r)
        res += atan2(e_target[1], e_target[0])
        if self.coords == COORD_GCS:
            res = pi / 2 - res
        return res


class HTargetFB(Feedback):
    """
    Control law heading towards target
    """

    def __init__(self, target: ndarray, coords: str):
        super().__init__(1, UniformWind(np.array((0., 0.))))
        self.target = np.zeros(2)
        self.target[:] = target
        self.coords = coords
        self.zero_ceil = 1e-3

    def value(self, t, x: ndarray):
        # Assuming GCS
        if self.coords == COORD_GCS:
            # Got to 3D cartesian assuming spherical earth
            lon, lat = x[0], x[1]
            # Vector normal to earth at position
            X3 = EARTH_RADIUS * np.array((cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)))
            e_phi = np.array((-sin(lon), cos(lon), 0.))
            e_lambda = np.array((-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)))
            lon, lat = self.target[0], self.target[1]
            X_target3 = EARTH_RADIUS * np.array((cos(lon) * cos(lat), sin(lon) * cos(lat), sin(lat)))
            e_target = np.zeros(2)
            e_target[0] = (X_target3 - X3) @ e_phi
            e_target[1] = (X_target3 - X3) @ e_lambda
        else:
            #  self.coords == COORD_CARTESIAN
            e_target = np.zeros(2)
            e_target[:] = self.target - x

        if np.linalg.norm(e_target) < self.zero_ceil:
            return 0.

        e_target = e_target / np.linalg.norm(e_target)
        res = atan2(e_target[1], e_target[0])
        if self.coords == COORD_GCS:
            res = pi / 2 - res
        return res


class WindAlignedFB(Feedback):
    """
    Defines a control law to align the airspeed vector with the wind vector
    """

    def __init__(self, wind):
        super().__init__(1, wind)

    def value(self, t, x):
        wind = self.wind.value(x)
        theta = np.arctan2(wind[1], wind[0])
        return theta


class FunFB(Feedback):

    def __init__(self, value_func, no_time=False):
        super().__init__(1, Wind())
        self._value_func = value_func
        self.no_time = no_time

    def value(self, t, x):
        if self.no_time:
            return self._value_func(x)
        else:
            return self._value_func(t, x)


class ConstantAS(AirspeedLaw):

    def __init__(self, airspeed):
        self.airspeed = airspeed

    def value(self, t, _):
        return self.airspeed


class ParamAS(AirspeedLaw):

    def __init__(self, aspd, t_end, t_start=0., interp='linear'):
        """
        Airspeed law interpreted as sampled functional data points
        at evenly spaced timestamps between t_start and t_end
        :param aspd: List of airspeed values in meters per second
        :param t_end: Duration of time window
        :param t_start: Start date if needed
        :param interp: Interpolation mode (only 'linear')
        """
        self.airspeed = np.zeros(aspd.shape)
        self.airspeed[:] = aspd
        self.t_end = t_end
        self.t_start = t_start
        self.interp = interp

    def _index(self, t):
        nt = self.airspeed.shape[0]
        if t <= self.t_start:
            return 0, 0.
        elif t >= self.t_end:
            return nt - 2, 1.
        tau = (t - self.t_start) / (self.t_end - self.t_start)
        k = int(tau * (nt - 1))
        return k, tau * (nt - 1) - k

    def value(self, t, _):
        k, alpha = self._index(t)
        return (1 - alpha) * self.airspeed[k] + alpha * self.airspeed[k+1]
