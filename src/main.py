import matplotlib as mpl
import numpy as np

from src.mermoz import MermozProblem
from src.model import Model5
from src.shooting import Shooting

mpl.style.use('seaborn-notebook')


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

    def u_vect(theta):
        return np.array([np.cos(theta), np.sin(theta)])

    omega = np.array([0.5, 0.3])
    gamma = -1.5
    flux = 1.
    T = 10.
    const_wind = np.array([-0.1, 0.])

    # model = Model1(v_a, v_w1, v_w2, x_f)
    model = Model5(v_a, x_f, const_wind, omega, gamma)
    mp = MermozProblem(model, T=T)
    list_p = list(map(lambda theta: u_vect(theta),
                      np.linspace(3 * np.pi / 4. + 1e-3, 5 * np.pi / 4. - 1e-3, 100)))
    # list_p.append(u_vect(0.8 * np.pi))
    # list_p.append(u_vect(0.79 * np.pi))
    # list_p.append(u_vect(0.78 * np.pi))
    for p in list_p:
        shoot = Shooting(model.dyn,
                         np.zeros(2),
                         T,
                         N_iter=100)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)
    list_headings = np.linspace(-0.6, 1.2, 10)
    # for heading in list_headings:
    #     mp.load_feedback(FixedHeadingFB(mp._model.wind, v_a, heading))
    #     mp.integrate_trajectory(TimedSC(T), int_step=0.01)
    mp.plot_trajs(color_mode="reachability")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
