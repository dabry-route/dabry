import random

import numpy as np
from tqdm import tqdm

from mermoz.wind import VortexWind, RadialGaussWind, RankineVortexWind, DoubleGyreWind, PointSymWind


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
        for i in tqdm(range(self.N_samples)):
            p = self.r_point()
            dwind = self.wind.d_value(p)
            dx = np.array([self.eps_x, 0.])
            dy = np.array([0., self.eps_y])

            dwind_est = np.column_stack((1 / (2 * self.eps_x) * (self.wind.value(p + dx) - self.wind.value(p - dx)),
                                         1 / (2 * self.eps_y) * (self.wind.value(p + dy) - self.wind.value(p - dy))))

            err_x = np.linalg.norm(dwind[:, 0] - dwind_est[:, 0])
            err_y = np.linalg.norm(dwind[:, 1] - dwind_est[:, 1])

            if err_x > self.rtol * self.wind_mean_norm or err_y > self.rtol * self.wind_mean_norm:
                print('Incorrect gradient')
                print(f'From function : {dwind}')
                print(f'    From test : {dwind_est}')
                failed += 1
            else:
                passed += 1

        print(f'Done. ({passed} passed, {failed} failed)')


if __name__ == '__main__':
    l_factor = 1e6
    x_min, x_max = -1. * l_factor, 3 * l_factor
    y_min, y_max = -1. * l_factor, 3 * l_factor
    x_center, y_center = 1. * l_factor, 1. * l_factor
    radius = 0.2 * l_factor
    # wind = VortexWind(x_center, y_center, radius)
    # wind = RankineVortexWind(x_center, y_center, 20., radius)
    # wind = PointSymWind(x_center, y_center, 20. / l_factor, 10/ l_factor)
    # wind = DoubleGyreWind(x_center, y_center, (x_max - x_min)/5., (y_max - y_min)/5., 20.)
    wind = RadialGaussWind(x_center, y_center, radius, 1/3 * np.log(0.5), 20.)

    tester = WindTester(x_min, x_max, y_min, y_max, wind)
    tester.setup()
    tester.test()
