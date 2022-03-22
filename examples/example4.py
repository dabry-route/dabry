import time

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from mermoz.feedback import FixedHeadingFB
from mermoz.problem import MermozProblem
from mermoz.model import ZermeloGeneralModel
from mermoz.rft import RFT
from mermoz.shooting import Shooting
from mermoz.stoppingcond import TimedSC
from mermoz.wind import VortexWind, UniformWind, RealWind, TSEqualWind

mpl.style.use('seaborn-notebook')


def example4():
    """
    Example of the shooting method on a vortex wind
    """
    print("Example 4")
    print("Building model... ", end='')
    t_start = time.time()
    # UAV airspeed in m/s
    v_a = 1.
    # UAV goal point x-coordinate in meters
    x_f = 1.
    # The time window upper bound in seconds
    T = 1. / v_a
    """
    # Precise the domain for the state. Avoid the center of vortices.
    def domain(x):
        # Box
        eps = 1e-3
        b4 = (0 - eps < x[0] < 1 + eps and -1 - eps < x[1] < 1. + eps)
        return b4

    uniform = UniformWind(np.array([5., 5.]))
    tsequal = TSEqualWind(10., -10., 1.)
    total_wind = 2 * RealWind('../../extractWindData/saved_wind/Vancouver-Honolulu-1.0')
    """
    vortex1 = VortexWind(0.5, 0.7, -1.)
    vortex2 = VortexWind(0.8, 0.2, -0.5)
    vortex3 = VortexWind(0.6, -0.4, 0.8)
    const_wind = UniformWind(np.array([-0.1, 0.]))

    # Precise the domain for the state. Avoid the center of vortices.
    def domain(x):
        # Vortices
        margin = 1e-1  # [m]
        b1 = np.linalg.norm(x - vortex1.omega) > margin
        b2 = np.linalg.norm(x - vortex2.omega) > margin
        b3 = np.linalg.norm(x - vortex3.omega) > margin

        # Box
        eps = 1e-3
        b4 = 0 - eps < x[0] < 1 + eps and -1 - eps < x[1] < 1. + eps
        return b1 and b2 and b3 and b4

        # Wind allows linear composition

    total_wind = 3. * const_wind + vortex1 + vortex2 + vortex3
    vortex1.value(np.array([0., 0.]))
    const_wind.value(np.array([0.3, 0.5]))
    total_wind.value(np.array([0.3, 0.5]))

    print(total_wind.value(np.array([0.5, 0.5])))

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, x_f)
    zermelo_model.update_wind(total_wind)

    # Initial point
    # x_init = np.array([0.45, 0.5])
    x_init = np.array([0.1, 0.1])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='only-map')
    mp.display.set_wind_density(3)
    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")

    nx = 80
    ny = 80
    nts = 50
    showstep = nts - 1
    delta_x = 1 / nx
    delta_t = 0.005
    print(delta_x / delta_t)

    print(f"Tracking reachability front ({nx}x{ny}, {nts}, dx=dy={delta_x:.2E}, dt={delta_t:.2E})... ", end='')
    t_start = time.time()

    rft = RFT(np.array([0., 0.]),
              1 / (nx - 1),
              1 / (ny - 1),
              delta_t,
              nx,
              ny,
              nts,
              mp,
              x_init)

    for k in range(nts - 1):
        rft.update_phi(method='sethian')

    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")
    rft.dump_rff('/home/bastien/Documents/work/mermoz/output/rff')

    # Set a list of initial adjoint states for the shooting method
    initial_headings = np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 20)
    list_p = list(map(lambda theta: -np.array([np.cos(theta), np.sin(theta)]), initial_headings))

    print(f"Shooting PMP trajectories ({len(list_p)})... ", end='')
    t_start = time.time()

    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined

    for k, p in enumerate(list_p):
        shoot = Shooting(zermelo_model.dyn, x_init, delta_t * showstep, adapt_ts=False, N_iter=100, domain=domain)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)

    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")
    """
    # Get also explicit control law traejctories
    # Here we add trajectories trying to steer the UAV
    # on a straight line starting from (0, 0) and with a given
    # heading angle
    list_headings = np.linspace(np.pi - 0.2, np.pi + 0.2, 1)
    print(f"Shooting explicit control laws ({list_headings.size})... ", end='')
    t_start = time.time()
    for heading in list_headings:
        mp.load_feedback(FixedHeadingFB(mp._model.wind, v_a, heading))
        mp.integrate_trajectory(x_init, TimedSC(T), int_step=0.05)
    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")
    """
    """
    fig, ax = plt.subplots(ncols=nts)
    for i in range(nts):
        # ax[i].imshow(rft.phi[:, :, i])
        ax[i].contourf(rft.phi[:, :, i], [-1e-2, 1e-2])
        ax[i].axis('equal')
    """

    t_start = time.time()
    print("Plotting trajectories... ", end='')
    mp.plot_trajs(color_mode="reachability")
    x_dak, y_dak = 0, 0
    x_nat, y_nat = 1, 1
    mp.display.map.scatter(x_dak, y_dak, s=8., color='red', marker='D')
    mp.display.map.scatter(x_nat, y_nat, s=8., color='red', marker='D')
    # mp.display.map.annotate('Dakar', (x_dak, y_dak), (0., -0.03))  # , textcoords='offset points')
    # mp.display.map.annotate('Natal', (x_nat, y_nat), xycoords='data')

    mp.display.map.set_ylim(-0.1, 1.1)
    X, Y = np.meshgrid(np.linspace(0., 1., nx), np.linspace(0., 1., ny))
    mp.display.map.contourf(Y, X, rft.phi[:, :, showstep], [-2e-3, 2e-3])

    t_end = time.time()
    print(f"Done ({t_end - t_start:.3f} s)")

    mp.display.set_title(
        rf'${nx}\times{ny},\; {nts},\; \Delta x=\Delta y={delta_x:.2E},\; dt={delta_t:.2E},\;  v_a={v_a},\; T={T:.3E}$')

    plt.show()


if __name__ == '__main__':
    example4()
