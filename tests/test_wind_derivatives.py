import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mermoz.wind import VortexWind, RadialGaussWind, RankineVortexWind, DoubleGyreWind, PointSymWind, DiscreteWind
from mermoz.misc import *


class WindTester:

    def __init__(self, x_min, x_max, y_min, y_max, wind, N_samples=10000, eps=1e-6, rtol=1e-6):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.wind = wind
        self.N_samples = N_samples
        self.eps_x = eps * (self.x_max - self.x_min)
        self.eps_y = eps * (self.y_max - self.y_min)

        self.rtol = rtol

        self.wind_mean = None
        self.wind_mean_norm = None

        self.seed = 42

    def r_point(self):
        x = self.x_min + (self.x_max - self.x_min) * random.random()
        y = self.y_min + (self.y_max - self.y_min) * random.random()
        return np.array([x, y])

    def setup(self):
        wind_samples = np.array([self.wind.value(self.r_point()) for _ in range(self.N_samples)])
        self.wind_mean = np.mean(wind_samples)
        self.wind_mean_norm = np.linalg.norm(self.wind_mean)

    def test(self):
        print(f'Launching tests (Seed : {self.seed})')
        random.seed(self.seed)
        passed = 0
        failed = 0
        points = np.zeros((self.N_samples, 2))
        c = np.zeros((self.N_samples, 4))
        red = np.array((1.0, 0., 0., 1.))
        green = np.array((0., 1.0, 0., 1.))
        for i in tqdm(range(self.N_samples)):
            p = self.r_point()
            points[i, :] = p
            dwind = self.wind.d_value(p)
            dx = np.array([self.eps_x, 0.])
            dy = np.array([0., self.eps_y])

            dwind_est = np.column_stack((1 / (2 * self.eps_x) * (self.wind.value(p + dx) - self.wind.value(p - dx)),
                                         1 / (2 * self.eps_y) * (self.wind.value(p + dy) - self.wind.value(p - dy))))

            err_x = np.linalg.norm(dwind[:, 0] - dwind_est[:, 0])
            err_y = np.linalg.norm(dwind[:, 1] - dwind_est[:, 1])

            if err_x > self.rtol * self.wind_mean_norm or err_y > self.rtol * self.wind_mean_norm:
                print('Incorrect gradient')
                print(f'At point :\n{p}')
                print(f'From function :\n{dwind}')
                print(f'    From test :\n{dwind_est}')
                c[i, :] = red
                failed += 1
            else:
                c[i, :] = green
                passed += 1

        print(f'Done. ({passed} passed, {failed} failed)')

        plt.scatter(points[:, 0], points[:, 1], s=3., c=c)
        plt.show()


if __name__ == '__main__':
    l_factor = 1e6
    x_min, x_max = -1. * l_factor, 3 * l_factor
    y_min, y_max = -1. * l_factor, 3 * l_factor
    x_center, y_center = 1. * l_factor, 1. * l_factor
    radius = 0.2 * l_factor
    # wind = VortexWind(x_center, y_center, radius)
    # wind = RankineVortexWind(x_center, y_center, 20., radius)
    # wind = PointSymWind(x_center, y_center, 20. / l_factor, 10/ l_factor)
    wind2 = DoubleGyreWind(x_center, y_center, (x_max - x_min)/5., (y_max - y_min)/5., 20.)
    # wind2 = RadialGaussWind(x_center, y_center, radius, 1/3 * np.log(0.5), 20.)
    wind = DiscreteWind(interp='linear')
    total_wind = DiscreteWind(force_analytical=True, interp='linear')
    total_wind.load('/home/bastien/Documents/work/mermoz/output/example_wf_san-juan_dublin/wind.h5')
    x_min, x_max = -1. * l_factor, 3 * l_factor
    y_min, y_max = -1. * l_factor, 3 * l_factor

    tester = WindTester(total_wind.x_min,
                        total_wind.x_max,
                        total_wind.y_min,
                        total_wind.y_max,
                        total_wind,
                        eps=1e-8,
                        rtol=1e-4)
    tester.setup()
    tester.test()
