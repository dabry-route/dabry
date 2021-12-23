import matplotlib.pyplot as plt
import numpy as np

def run():
    l = 3.
    def f(x, l):
        return -l * np.sin(2*x)

    for x0 in [0.1, 1., np.pi / 2. + 1e-2, np.pi / 2. + 1e-3]:
        t = 0.
        t_list = [t]
        dt = 0.001
        x = x0
        x_list = [x]
        for i in range(5000):
            x = x + f(x, l)*dt
            x_list.append(x)
            t += dt
            t_list.append(t)

        plt.plot(t_list, x_list, label=f'$x_0=${x0}')
    # atan = []
    # for t in t_list:
    #     atan.append(3* np.pi /4. + 0.5 * np.arctan(3.*(t-2.33)))
    # plt.plot(t_list, atan, label=f'arctan')
    tanh = []
    for t in t_list:
        tanh.append(3* np.pi /4. + np.pi/4 * np.tanh(1.85*(t-2.3)))
    plt.plot(t_list, tanh, label=f'tanh')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run()