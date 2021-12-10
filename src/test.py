import random

import matplotlib as mpl
import numpy as np

from src.feedback import FixedHeadingFB
from src.mermoz import MermozProblem
from src.model import ZermeloGeneralModel
from src.shooting import Shooting
from src.stoppingcond import TimedSC
from src.trajectory import Trajectory, AugmentedTraj
from wind import VortexWind, UniformWind

mpl.style.use('seaborn-notebook')


def example1():
    """
    Example of the shooting method on a vortex wind
    """
    # UAV airspeed in m/s
    v_a = 1.
    # UAV goal point x-coordinate in meters
    x_f = 1.
    # The time window upper bound in seconds
    T = 0.08
    # The vortex center definition in meters
    omega = np.array([0.5, 0.3])
    # The vortex intensity in m^2/s
    gamma = -1.5

    center1 = 0.5, 0.7
    center2 = 0.8, 0.2
    center3 = 0.6, -0.4
    gamma1, gamma2, gamma3 = -1., -0.5, 0.8
    vortex1 = VortexWind(center1[0], center1[1], gamma1)
    vortex2 = VortexWind(center2[0], center2[1], gamma2)
    vortex3 = VortexWind(center3[0], center3[1], gamma3)
    const_wind = UniformWind(np.array([-0.1, 0.]))

    # Wind allows linear composition
    total_wind = 3. * const_wind + vortex1 + vortex2 + vortex3

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a, x_f)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = np.array([0.5, 0.1])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='full-adjoint')

    dl = 1e-5
    dx = np.array([dl, 0.])
    dy = np.array([0., dl])
    N_points = 1000

    def hm_d_value(x):
        return np.hstack(((mp._model.wind.value(x + dx) - mp._model.wind.value(x)) / dl,
                          (mp._model.wind.value(x + dy) - mp._model.wind.value(x)) / dl)).reshape((2, 2))

    def order1(x, vw, grad_vw, t, s):
        return x + vw * t + t * v_a * s

    def order2(x, vw, grad_vw, t, s):
        lie = grad_vw.dot(vw)
        return order1(x, vw, grad_vw, t, s) + 0.5 * lie * t ** 2 + t ** 2 / 2. * np.einsum('i,ij,j', s, grad_vw, s) * s

    unit_sphere = list(map(lambda theta: np.array([np.cos(theta), np.sin(theta)]),
                           np.linspace(-np.pi + 1e-3, np.pi - 1e-3, N_points)))

    vw = mp._model.wind.value(x_init)
    grad_vw = mp._model.wind.d_value(x_init)
    # for f in [order1, order2]:
    t = T / 2
    """
    points = np.zeros((N_points, 2))
    for k, s in enumerate(unit_sphere):
        points[k, :] = order1(x_init, vw, grad_vw, t, s)
    points = points[::N_points // 4]
    mp.trajs.append(Trajectory(np.zeros(len(points)), points, np.zeros(len(points)), len(points), type="point"))
    for x in points:
        vw = mp._model.wind.value(x)
        grad_vw = mp._model.wind.d_value(x)
        new_points = np.zeros((N_points, 2))
        for k, s in enumerate(unit_sphere):
            new_points[k, :] = order1(x, vw, grad_vw, T / 2., s)
        mp.trajs.append(Trajectory(np.zeros(len(new_points)), new_points, np.zeros(len(new_points)), len(new_points),
                                   type="approx"))
    """
    t = T
    points = np.zeros((N_points, 2))
    for k, s in enumerate(unit_sphere):
        points[k, :] = order1(x_init, vw, grad_vw, t, s)
    # mp.trajs.append(Trajectory(np.zeros(len(points)), points, np.zeros(len(points)), len(points), type="approx"))


    # mp.trajs.append(Trajectory(np.zeros(1), x_init.reshape((1, 2)), np.zeros(1), 1))

    # Set a list of initial adjoint states for the shooting method
    initial_headings = np.linspace(- np.pi + 1e-3, np.pi - 1e-3, 100)
    list_p = list(map(lambda theta: -np.array([np.cos(theta), np.sin(theta)]), initial_headings))

    X, Y = np.meshgrid(np.linspace(-0.05, 1.05, 50), np.linspace(-1.05, 1.05, 50))

    cartesian = np.dstack((X, Y)).reshape(-1, 2)

    def sample_sphere(center, radius, N_samples=1000):
        samples = np.zeros((N_samples, 2))
        for k in range(N_samples):
            samples[k, :] = center + unit_sphere[random.randint(0, len(unit_sphere)-1)] * random.random() * radius
        return samples

    print(cartesian)
    delete_index = []
    for k in range(cartesian.shape[0]):
        to_delete = False
        if np.linalg.norm(cartesian[k, :] - center1) < 0.15:
            to_delete = True
        if np.linalg.norm(cartesian[k, :] - center2) < 0.15:
            to_delete = True
        if np.linalg.norm(cartesian[k, :] - center3) < 0.15:
            to_delete = True
        if to_delete:
            delete_index.append(k)
    cartesian = np.delete(cartesian, delete_index, axis=0)
    print(cartesian)
    mp.display.setup()
    # mp.display.map.scatter(cartesian[:, 0], cartesian[:, 1])

    list_grads = np.array(list(map(lambda x: mp._model.wind.d_value(x), cartesian)))
    list_norms = np.array(list(map(lambda x: np.linalg.norm(mp._model.wind.value(x)), cartesian)))
    print(list_norms)
    V_v = np.max(list_norms)
    print(f'V_v : {V_v}')
    new_V_v = 0.
    while np.abs(V_v - new_V_v) > 1e-3:
        V_v = new_V_v
        N_samples = 1000
        sphere = sample_sphere(x_init, T*(v_a + V_v), N_samples=N_samples)
        list_norms = np.array(list(map(lambda x: np.linalg.norm(mp._model.wind.value(x)), sphere)))
        print(f'V_v : {V_v}')
        print(f'new_V_v : {new_V_v}')
        new_V_v = np.max(list_norms)
    print(f'Converged V_v : {new_V_v}')

    list_grads = np.array(list(map(lambda x: mp._model.wind.d_value(x), sample_sphere(x_init, T*(v_a + V_v)))))
    N_v = np.max(list_grads)
    print(f'N_v : {N_v}')
    print(f'V_v : {V_v}')
    print(f'Trivial lb : {1./(v_a + V_v)}')

    def enlarge(points, factor):
        res = np.zeros(points.shape)
        N_points = points.shape[0]
        for k in range(N_points):
            tangent = np.zeros(2)
            tangent[:] = points[(k - 1) % N_points, :] - points[(k + 1) % N_points, :]
            tangent[:] = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            res[k, :] = points[k] + normal * factor
        return res

    new_points = enlarge(points, T**2/2. * (v_a + V_v)*N_v)
    mp.trajs.append(Trajectory(np.zeros(len(new_points)), new_points, np.zeros(len(new_points)), len(new_points), type="approx"))

    fast_approx = np.zeros((N_points, 2))
    for k, s in enumerate(unit_sphere):
        fast_approx[k, :] = x_init + T*(v_a + V_v)*s
    mp.trajs.append(Trajectory(np.zeros(len(fast_approx)), fast_approx, np.zeros(len(fast_approx)), len(fast_approx), type="integral"))


    grad_vw = mp._model.wind.d_value(x_init)
    print(f'grad_vw :\n{grad_vw}')
    _, eigvecs = np.linalg.eig(grad_vw)
    print(f'eigvecs :\n{eigvecs}')
    e1 = eigvecs[:, 0]
    print(e1)
    mp.display.map_adjoint.arrow(0., 0., e1[0], e1[1])

    # Clover shape
    """
    accel = np.zeros((N_points, 2))
    for k, s in enumerate(unit_sphere):
        accel[k, :] = np.einsum('i,ij,j', s, grad_vw, s) * s
    mp.trajs.append(AugmentedTraj(np.zeros(N_points), np.zeros((N_points, 2)), accel, np.zeros(N_points), N_points))
    """
    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined

    for p in list_p:
        shoot = Shooting(zermelo_model.dyn, x_init, T, N_iter=100)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        mp.trajs.append(aug_traj)

    # Get also explicit control law traejctories
    # Here we add trajectories trying to steer the UAV
    # on a straight line starting from (0, 0) and with a given
    # heading angle
    """
    list_headings = np.linspace(-0.6, 1.2, 10)
    for heading in list_headings:
        mp.load_feedback(FixedHeadingFB(mp._model.wind, v_a, heading))
        mp.integrate_trajectory(x_init, TimedSC(T), int_step=0.01)
    """
    mp.plot_trajs(color_mode="monocolor")


if __name__ == '__main__':
    example1()
