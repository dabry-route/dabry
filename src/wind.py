import math

import numpy as np
from numpy import ndarray


class Wind:

    def __init__(self, value_func=None, d_value_func=None, descr=''):
        """
        Builds a windfield which is a smooth vector field of space
        value_func Function taking a point in space (ndarray) and returning the
        wind value at the given point.
        d_value_func Function taking a point in space (ndarray) and returning the
        jacobian of the windfield at the given point.
        """
        self.value = value_func
        self.d_value = d_value_func
        self.descr = descr

    def __add__(self, other):
        """
        Add windfields

        :param other: Another windfield
        :return: The sum of the two windfields
        """
        if not isinstance(other, Wind):
            raise TypeError(f"Unsupported type for addition : {type(other)}")
        return Wind(value_func=lambda x: self.value(x) + other.value(x),
                    d_value_func=lambda x: self.d_value(x) + other.d_value(x),
                    descr=self.descr + ', ' + other.descr)

    def __mul__(self, other):
        """
        Handles the scaling of a windfield by a real number

        :param other: A real number (float)
        :return: The scaled windfield
        """
        if not isinstance(other, float):
            raise TypeError(f"Unsupported type for multiplication : {type(other)}")
        return Wind(value_func=lambda x: other * self.value(x),
                    d_value_func=lambda x: other * self.d_value(x),
                    descr=self.descr)

    def __rmul__(self, other):
        return self.__mul__(other)

    def e_minus(self, x):
        """
        Returns an eigenvector of the vector space associated to the
        negative eigenvalue of the jacobian of the windfield

        :param x: The point in space where the eigenvector is computed
        :return: The eigenvector
        """
        d_wind = self.d_value(x)
        dw_x = d_wind[0, 0]
        dw_y = d_wind[0, 1]
        lambda_0 = np.sqrt(dw_x ** 2 + dw_y ** 2)
        return np.array([- dw_y, lambda_0 + dw_x])

    def grad_norm(self, x):
        """
        Returns the gradient of the wind norm function at given point

        :param x: The point at which to compute the gradient
        :return: The gradient
        """
        wind = self.value(x)
        return np.linalg.norm(self.d_value(x).transpose().dot(wind / np.linalg.norm(wind)))

    def __str__(self):
        return self.descr


class TwoSectorsWind(Wind):

    def __init__(self,
                 v_w1: float,
                 v_w2: float,
                 x_switch: float):
        """
        Wind configuration where wind is constant over two half-planes separated by x = x_switch. x-wind is null.

        :param v_w1: y-wind value for x < x_switch
        :param v_w2: y-wind value for x >= x_switch
        :param x_switch: x-coordinate for sectors separation
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.v_w1 = v_w1
        self.v_w2 = v_w2
        self.x_switch = x_switch
        self.descr = f'Two sectors' #({self.v_w1:.2f} m/s, {self.v_w2:.2f} m/s)'

    def value(self, x):
        return np.array([0, self.v_w1 * np.heaviside(self.x_switch - x[0], 0.)
                         + self.v_w2 * np.heaviside(x[0] - self.x_switch, 0.)])

    def d_value(self, x):
        return np.array([[0, 0],
                         [0, 0]])


class TSEqualWind(TwoSectorsWind):

    def __init__(self, v_w1, v_w2, x_f):
        """
        TwoSectorsWind but the sector separation is midway to the target

        :param x_f: Target x-coordinate.
        """
        super().__init__(v_w1, v_w2, x_f / 2)


class UniformWind(Wind):

    def __init__(self, wind_vector: ndarray):
        """
        :param wind_vector: Direction and strength of wind
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.wind_vector = wind_vector
        self.descr = f'Uniform' #({np.linalg.norm(self.wind_vector):.2f} m/s, {math.floor(180 / np.pi * np.arctan2(self.wind_vector[1], self.wind_vector[0]))})'

    def value(self, x):
        return self.wind_vector

    def d_value(self, x):
        return 0.


class VortexWind(Wind):

    def __init__(self,
                 x_omega: float,
                 y_omega: float,
                 gamma: float):
        """
        A vortex from potential theory

        :param x_omega: x_coordinate of vortex center in m
        :param y_omega: y_coordinate of vortex center in m
        :param gamma: Circulation of the vortex in m^2/s. Positive is ccw vortex.
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)
        self.x_omega = x_omega
        self.y_omega = y_omega
        self.omega = np.array([x_omega, y_omega])
        self.gamma = gamma
        self.descr = f'Vortex' #({self.gamma:.2f} m^2/s, at ({self.x_omega:.2f}, {self.y_omega:.2f}))'

    def value(self, x):
        r = np.linalg.norm(x - self.omega)
        e_theta = np.array([-(x - self.omega)[1] / r,
                            (x - self.omega)[0] / r])
        return self.gamma / (2 * np.pi * r) * e_theta

    def d_value(self, x):
        r = np.linalg.norm(x - self.omega)
        x_omega = self.x_omega
        y_omega = self.y_omega
        return self.gamma / (2 * np.pi * r ** 4) * \
               np.array([[2 * (x[0] - x_omega) * (x[1] - y_omega), (x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2],
                         [(x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2, -2 * (x[0] - x_omega) * (x[1] - y_omega)]])


class SourceWind(Wind):

    def __init__(self,
                 x_omega: float,
                 y_omega: float,
                 flux: float):
        """
        A source from potentiel theory

        :param x_omega: x_coordinate of source center in m
        :param y_omega: y_coordinate of source center in m
        :param flux: Flux of the source m^2/s. Positive is source, negative is well.
        """
        super().__init__()
        self.x_omega = x_omega
        self.y_omega = y_omega
        self.omega = np.array([x_omega, y_omega])
        self.flux = flux

    def value(self, x):
        r = np.linalg.norm(x - self.omega)
        e_r = (x - self.omega) / r
        return self.flux / (2 * np.pi * r) * e_r

    def d_value(self, x):
        raise ValueError("No derivative implemented for source wind")


class LinearWind(Wind):
    """
    This wind is NOT a potential flow
    """

    def __init__(self, gradient: ndarray, origin: ndarray, value_origin: ndarray):
        """
        :param gradient: The wind gradient
        :param origin: The origin point for the origin value
        :param value_origin: The origin value
        """
        super().__init__(value_func=self.value, d_value_func=self.d_value)

        self.gradient = np.zeros(gradient.shape)
        self.origin = np.zeros(origin.shape)
        self.value_origin = np.zeros(value_origin.shape)

        self.gradient[:] = gradient
        self.origin[:] = origin
        self.value_origin[:] = value_origin

    def value(self, x):
        return self.gradient.dot(x - self.origin) + self.value_origin

    def d_value(self, x):
        return self.gradient
