import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_min, x_max = 0., 1.
    y_min, y_max = 0., 1.
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(x_min, x_max, 100))
    nx = 5
    ny = 5
    u = np.array([[random.randint(-5, 5)/5. for _ in range(ny)] for _ in range(nx)], dtype=float)
    print(u.shape)


    def f(x, y):
        xx = (x - x_min) / (x_max - x_min)
        yy = (y - y_min) / (y_max - y_min)
        delta_x = (x_max - x_min) / (nx - 1)
        delta_y = (y_max - y_min) / (ny - 1)
        i = int((nx - 1) * xx)
        if i == nx - 1:
            i -= 1
        j = int((ny - 1) * yy)
        if j == ny - 1:
            j -= 1
        # Tile bottom left corner x-coordinate
        xij = delta_x * i + x_min
        # Point relative position in tile
        a = (x - xij) / delta_x
        # Tile bottom left corner y-coordinate
        yij = delta_y * j + y_min
        b = (y - yij) / delta_y
        try:
            return (1 - b) * ((1 - a) * u[i, j] + a * u[i + 1, j]) + b * ((1 - a) * u[i, j + 1] + a * u[i + 1, j + 1])
        except IndexError:
            print(i, j)
            return 0.


    Z = np.zeros(X.shape)
    for ki in range(X.shape[0]):
        for kj in range(X.shape[1]):
            Z[ki, kj] = f(X[ki, kj], Y[ki, kj])

    ax.plot_wireframe(X, Y, Z)
    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    for i in range(nx):
        for j in range(ny):
            delta_x = (x_max - x_min) / (nx - 1)
            delta_y = (y_max - y_min) / (ny - 1)
            ax.scatter(i * delta_x, j * delta_y, u[i, j], color='red')
    plt.show()
