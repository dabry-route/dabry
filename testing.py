import matplotlib.pyplot as plt
import numpy as np
from math import atan, tan, exp


def run():
    lambda_0 = 2.
    l = lambda_0 / 2.

    def f(x, l):
        return -l * np.sin(2 * x)

    for theta_0 in [0.1, 1., np.pi / 2. + 1e-2, np.pi / 2. + 1e-3]:
        t = 0.
        t_list = [t]
        dt = 0.001
        theta = theta_0
        theta_list = [theta]
        for i in range(5000):
            theta = theta + f(theta, l) * dt
            theta_list.append(theta)
            t += dt
            t_list.append(t)

        # plt.plot(t_list, theta_list, label=f'$theta_0=${theta_0:.2f}')

    def model(theta_0, t, lambda_0):
        const = 0.
        if theta_0 < -np.pi / 2.:
            const = -np.pi
        if np.pi / 2. < theta_0:
            const = np.pi
        return const + atan(tan(theta_0) * exp(-lambda_0 * t))

    fig, ax = plt.subplots()
    for theta_0 in [0., 1., 1.5, 1.6, 2., 3.]:
        model_list = []
        for t in t_list:
            model_list.append(model(theta_0, t, lambda_0))
        ax.plot(t_list, model_list, label=rf'$\theta_0=${theta_0:.2f}')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\theta$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run()
