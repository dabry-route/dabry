import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import portion as ivl
from matplotlib import mlab

from src.feedback import ZermeloPMPFB, ConstantFB
from src.mermoz import MermozProblem
from src.model import Model1, Model2, Model3
from src.shooting import Shooting
from src.stoppingcond import TimedSC
from wind import TwoSectorsWind

import matplotlib as mpl

mpl.style.use('seaborn-notebook')


def f1(x, y):
    return np.sin(2 * np.pi * x) * np.cos(3 * 2 * np.pi * y)


# def f2(x, u, py, va, vw, l):
#     return np.sin(u) + py * va / (1 + py * vw_fun(x, vw, l))

class TrajProb:

    def __init__(self, va, vw1, vw2, py, l, N_traj=10000, dt=0.0005, linear_wind=False):
        self.va = va
        self.vw1 = vw1
        self.vw2 = vw2
        self.r1 = vw1 / va
        self.r2 = vw2 / va
        self.py = py
        self.l = l
        self.N_traj = N_traj
        self.N_sig = 0
        self.dt = dt
        self.traj = np.zeros((self.N_traj, 2))
        self.linear_wind = linear_wind

    def domain(self):
        r1, r2 = self.r1, self.r2
        res = ivl.open(-ivl.inf, ivl.inf)
        for r in [r1, r2]:
            if r > 1:
                res = res.intersection(ivl.closedopen(-1 / (1 + r), ivl.inf))
            elif r > -1:
                res = res.intersection(ivl.closed(-1 / (1 + r), 1 / (1 - r)))
            else:
                res = res.intersection(ivl.openclosed(-ivl.inf, 1 / (1 - r)))
        return res

    def wind_value(self, x):
        if self.linear_wind:
            return self.vw1 * x / self.l
        else:
            return self.vw1 * np.heaviside(self.l / 3 - x, 0.) + self.vw2 * np.heaviside(x - self.l / 3, 0.)

    def dyn(self, x, u):
        return np.array([self.va * np.cos(u), self.va * np.sin(u) + self.wind_value(x[0])])

    def u_feedback(self, x):
        wv = self.wind_value(x[0])
        sin_value = -self.py * self.va / (1 + self.py * wv)
        if sin_value > 1.:
            raise ValueError("u_feedback : sin value above 1")
        elif sin_value < -1.:
            raise ValueError("u_feedback : sin value below -1")
        return np.arcsin(sin_value)

    def compute_analytic(self):
        self.theta_1 = self.theta(self.u_1)

    def integrate(self):
        self.t_f = 0.
        i = 0
        x = self.traj[0]
        self.u_1 = self.u_feedback(x)
        self.theta_1 = self.theta(self.u_1)
        while x[0] < self.l:
            try:
                self.traj[i + 1] = x + self.dt * self.dyn(x, self.u_feedback(x))
            except IndexError:
                raise IndexError(f"Iteration limit {self.N_traj} reached in integration")
            x = self.traj[i + 1]
            i += 1
            self.t_f += self.dt
        self.N_sig = i + 1
        self.u_2 = self.u_feedback(x)
        self.theta_2 = self.theta(self.u_2, index=2)
        self.delta_y = self.l / 2 * (np.tan(self.theta_1) + np.tan(self.theta_2))
        self.y_f = x[1]

    def theta(self, u, index=1):
        if index not in [1, 2]:
            raise ValueError('Unknown theta index : {}'.format(index))
        vw = self.vw1 if index == 1 else self.vw2
        return np.arcsin((vw + self.va * np.sin(u)) / np.sqrt(
            vw ** 2 + self.va ** 2 + 2 * vw * self.va * np.sin(u)))

    def theta_true(self, second=False):
        if self.N_sig == 0:
            raise ValueError("Trajectory was not integrated")
        delta_y = np.max(self.traj, axis=0)[1]
        return np.arctan(2 * delta_y / self.l)

    def residual_term(self, py, vw):
        return (vw + py * (vw ** 2 - self.va ** 2)) / np.sqrt(
            (1 + py * vw) * (self.va ** 2 + vw ** 2 + py * vw * (vw ** 2 - self.va ** 2)))

    def residual(self, py):
        return self.residual_term(py, self.vw1) + self.residual_term(py, self.vw2)

    def travel_time(self):
        r1, r2, l, v_a, p_y = self.r1, self.r2, self.l, self.va, self.py
        mu = v_a * p_y
        return l / (2 * v_a) * (np.abs(1 + mu * r1) / (np.sqrt(1 + 2 * mu * r1 + mu ** 2 * (r1 ** 2 - 1)))
                                + np.abs(1 + mu * r2) / (np.sqrt(1 + 2 * mu * r2 + mu ** 2 * (r2 ** 2 - 1))))

    def plot(self, fig, ax, opt=False):
        ax.set_xlim(-.1, 1.1)
        ax.set_ylim(-1.5, 1.)
        label = r'$p_y$ = {py:.3f}, $t_f$ = {t_f:.3f}$s$, $u_1$ = {t1:.1f}$deg$, $u_2$ = {t2:.1f}$deg$, $\theta_1$ = {theta_1:.2f}, $\theta_2$ = {theta_2:.2f}, $\Delta y$ = {delta_y:.2f}, $\Delta y - y_f$ = {diff:.2f}'.format(
            py=self.py,
            t_f=self.t_f,
            t1=self.u_1 * 180 / np.pi,
            t2=self.u_2 * 180 / np.pi,
            theta_1=self.theta_1 * 180 / np.pi,
            theta_2=self.theta_2 * 180 / np.pi,
            delta_y=self.delta_y,
            diff=self.delta_y - self.y_f)
        ax.scatter(self.traj[:self.N_sig, 0], self.traj[:self.N_sig, 1], s=0.5, label=label,
                   marker=('x' if opt else None))

    def display_wind(self, fig, ax, nx=8, ny=10):

        X, Y = np.meshgrid(np.linspace(0.1, 0.9, nx), np.linspace(-1, 1, ny))

        U = np.zeros(nx * ny)
        V = self.wind_value(X)

        ax.quiver(X, Y, U, V, color='blue', alpha=0.3, headwidth=3, width=0.005, scale=15)


def run():
    v_a = 1.
    x_f = 1.
    v_w1 = 0.5
    kappa = 0.
    v_w2 = kappa * v_w1
    p_y = 0.

    # model = Model1(v_a, v_w1, v_w2, x_f)
    # mp = MermozProblem(model)
    # mp.load_feedback(ZermeloPMPFB(model.wind, v_a, p_y))
    # mp.integrate_trajectory()
    # print(mp.trajs[0].points)
    # mp.plot_trajs()
    # print(mp.trajs[0].get_final_time())
    # exit(1)

    omega = np.array([0.5, 0.3])
    gamma = -2.
    flux = 1.

    # model = Model1(v_a, v_w1, v_w2, x_f)
    model = Model2(v_a, x_f, omega, gamma)
    mp = MermozProblem(model)
    list_p = list(map(lambda theta: np.array([np.cos(theta), np.sin(theta)]),
                      np.linspace(np.pi/2. + 1e-3, 3*np.pi/2. - 1e-3, 20)))
    for p in list_p:
        shoot = Shooting(model.dyn,
                         np.zeros(2),
                         1.,
                         N_iter=100)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)
    mp.plot_trajs(mode="reachability")

    exit(1)
    # model = Model1(v_a, v_w1, v_w2, x_f)
    # model = Model2(v_a, x_f, omega, gamma)
    model = Model3(v_a, x_f, np.array([0.5, 0]), flux)
    mp = MermozProblem(model)
    # mp.reachability(1., N_samples=200, int_step=0.01)
    mp.load_feedback(ConstantFB(0.1))
    mp.integrate_trajectory(TimedSC(1.), int_step=0.01)
    mp.plot_trajs()
    exit(1)
    mp.load_feedback(ConstantFB(0.))
    mp.integrate_trajectory()
    print(mp.trajs[0].points)
    mp.plot_trajs()
    print(mp.trajs[0].get_final_time())
    exit(1)

    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.axvline(x=1., color='k', linewidth=0.5)

    # for py in [0., .1, .3]:
    #     tp = TrajProb(va, vw1, vw2, py, l, linear_wind=True)
    #     tp.display_wind(fig, ax)
    #     tp.integrate()
    #     tp.plot(fig, ax)
    # plt.legend()
    # plt.show()
    # exit(1)

    tp = TrajProb(va, vw1, vw2, py, l)

    py_star = scipy.optimize.newton(tp.residual, py)
    print(f"py_star = {py_star}")
    tp_star = TrajProb(va, vw1, vw2, py_star, l)
    u1 = tp_star.u_feedback(np.array([0, 0]))
    u2 = tp_star.u_feedback(np.array([l, 0]))
    theta_1_star = tp_star.theta(u1, index=1)
    theta_2_star = tp_star.theta(u2, index=2)
    tp_star.integrate()
    print(f'theta_1_star : {theta_1_star}')
    print(f'theta_2_star : {theta_2_star}')
    print(f'theta_true : {tp_star.theta_true()}')
    print(f'u1 : {u1}')
    print(f'u2 : {u2}')
    print(f'T : {tp_star.travel_time()}')
    tp_star.plot(fig, ax)
    plt.show()

    dom = tp.domain()
    print(dom)
    lo, up = dom.lower, dom.upper
    for k, py in enumerate(set([py_star, 0.]).union(set(np.linspace(lo + 0.1 * np.abs(lo), up - 0.1 * np.abs(up), 5)))):
        print(f"py : {py}")
        tp = TrajProb(va, vw1, vw2, py, l)
        tp.integrate()
        tp.plot(fig, ax, opt=not k)
        print(f"    Analytic theta_1 : {tp.theta_1}")
        print(f"    Numerical theta_1 : {tp.theta_true()}")
        print(f"    Analytic theta_2 : {tp.theta_2}")
        print(f"    Residual : {tp.residual(tp.py)}")
    ax.grid(True)
    ax.legend(fontsize=6)
    plt.show()

    # fig, ax = plt.subplots()
    #
    # X, U = np.meshgrid(np.linspace(-.1, 1.1, 150), np.linspace(-np.pi / 2, np.pi / 2, 150))
    # Z = f2(X, U, py, va, vw, l)
    # c = ax.pcolormesh(X, U, Z, cmap='viridis', shading='auto', vmin=np.min(Z), vmax=np.max(Z))
    #
    # fig.colorbar(c)
    #
    # ax.set_xlabel('x ($m$)')
    # ax.set_ylabel('u ($rad$)')
    #
    # plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
