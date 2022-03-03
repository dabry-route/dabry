import numpy as np
from numpy import ndarray

from src.mermoz import MermozProblem


def upwind_diff(field, axis, delta):
    if axis not in [0, 1]:
        print("Only handles 2D fields")
        exit(1)
    nx, ny = field.shape
    res = np.zeros(field.shape)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if axis == 0:
                if field[i - 1, j] < field[i, j] < field[i + 1, j]:
                    res[i, j] = (field[i, j] - field[i - 1, j]) / delta
                elif field[i - 1, j] > field[i, j] > field[i + 1, j]:
                    res[i, j] = (field[i + 1, j] - field[i, j]) / delta
                elif field[i - 1, j] <= field[i, j] >= field[i + 1, j]:
                    res[i, j] = (field[i + 1, j] - field[i - 1, j]) / (2 * delta)
                elif field[i - 1, j] >= field[i, j] <= field[i + 1, j]:
                    res[i, j] = 0
            else:
                # axis == 1
                if field[i, j - 1] < field[i, j] < field[i, j + 1]:
                    res[i, j] = (field[i, j] - field[i, j - 1]) / delta
                elif field[i, j - 1] > field[i, j] > field[i, j + 1]:
                    res[i, j] = (field[i, j + 1] - field[i, j]) / delta
                elif field[i, j - 1] <= field[i, j] >= field[i, j + 1]:
                    res[i, j] = (field[i, j + 1] - field[i, j - 1]) / (2 * delta)
                elif field[i, j - 1] >= field[i, j] <= field[i, j + 1]:
                    res[i, j] = 0
    for j in range(1, field.shape[1] - 1):
        res[0, j] = res[1, j]
        res[nx - 1, j] = res[nx - 2, j]
    for i in range(1, field.shape[0] - 1):
        res[i, 0] = res[i, 1]
        res[i, ny - 1] = res[i, ny - 2]
    res[0, 0] = res[1, 1]
    res[nx - 1, 0] = res[nx - 2, 1]
    res[0, ny - 1] = res[1, ny - 2]
    res[nx - 1, ny - 1] = res[nx - 2, ny - 2]
    return res


class RFT:
    """
    Reachability front tracker
    """

    def __init__(self, origin, nx, ny, n_ts, mp : MermozProblem):
        self.mp = mp
        self.origin = np.zeros(2)
        self.origin[:] = origin
        self.wind = np.zeros((nx, ny, n_ts, 2))
        for i in range(nx):
            for j in range(ny):
                for k in range(n_ts):
                    self.wind[i, j, k, :] = self.mp._model.wind.value()
        self.phi = np.zeros((nx, ny, n_ts))
        self.t = 0

    def update_phi(self, delta_t, delta_x, delta_y, speed):
        phi_star = np.zeros(self.phi.shape)
        phi_star[:] = delta_t / 2 * (self.phi - speed * np.sqrt(
            upwind_diff(self.phi, 0, delta_x) ** 2 + upwind_diff(self.phi, 1, delta_y) ** 2))
        phi_sstar = np.zeros(self.phi.shape)
        # phi_sstar[:] = delta_t * (phi_star - )

    def time_to_go(self,
                   x_init: ndarray,
                   x_end: ndarray):
        pass
