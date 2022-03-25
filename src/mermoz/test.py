import random
import matplotlib as mpl
import numpy as np

from .feedback import FixedHeadingFB
from .problem import MermozProblem
from .model import ZermeloGeneralModel
from .shooting import Shooting
from .stoppingcond import TimedSC
from .trajectory import Trajectory, AugmentedTraj
from .wind import VortexWind, UniformWind

mpl.style.use('seaborn-notebook')


def example1():
    """
    Example of the shooting method on a vortex wind
    """
    factor = 1.
    # UAV airspeed in m/s
    v_a = factor * 1.
    # UAV goal point x-coordinate in meters
    x_f = 1.
    # The time window upper bound in seconds
    T = 0.8
    # The vortex center definition in meters
    omega = np.array([0.5, 0.3])
    # The vortex intensity in m^2/s
    gamma = -1.5

    N_shooting_iter = 500

    center1 = 0.5, 0.7
    center2 = 0.8, 0.2
    center3 = 0.6, -0.4
    gamma1, gamma2, gamma3 = 1., 0.5, -0.8
    vortex1 = VortexWind(center1[0], center1[1], gamma1 * factor)
    vortex2 = VortexWind(center2[0], center2[1], gamma2 * factor)
    vortex3 = VortexWind(center3[0], center3[1], gamma3 * factor)
    const_wind = UniformWind(factor * np.array([-0.1, 0.]))

    # Wind allows linear composition
    total_wind = 3. * const_wind + vortex1 + vortex2 + vortex3

    # Creates the cinematic model
    zermelo_model = ZermeloGeneralModel(v_a)
    zermelo_model.update_wind(total_wind)

    # Initial point
    x_init = np.array([0., 0.])

    # Creates the navigation problem on top of the previous model
    mp = MermozProblem(zermelo_model, T=T, visual_mode='full')

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
                           np.linspace(0. + 1e-3, 2 * np.pi - 1e-3, N_points)))

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
    # initial_headings = np.linspace(- 3 * np.pi / 4 + 1e-3, 3 * np.pi / 4 - 1e-3, 100)
    initial_headings = np.linspace(-np.pi / 4,  np.pi / 4, 50)
    list_p = list(map(lambda theta: -np.array([np.cos(theta), np.sin(theta)]), initial_headings))

    X, Y = np.meshgrid(np.linspace(-0.05, 1.05, 50), np.linspace(-1.05, 1.05, 50))

    cartesian = np.dstack((X, Y)).reshape(-1, 2)

    def sample_sphere(center, radius, N_samples=1000):
        samples = np.zeros((N_samples, 2))
        for k in range(N_samples):
            samples[k, :] = center + unit_sphere[random.randint(0, len(unit_sphere) - 1)] * random.random() * radius
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
    # while np.abs(V_v - new_V_v) > 1e-3:
    #     V_v = new_V_v
    #     N_samples = 1000
    #     sphere = sample_sphere(x_init, T*(v_a + V_v), N_samples=N_samples)
    #     list_norms = np.array(list(map(lambda x: np.linalg.norm(mp._model.wind.value(x)), sphere)))
    #     print(f'V_v : {V_v}')
    #     print(f'new_V_v : {new_V_v}')
    #     new_V_v = np.max(list_norms)
    print(f'Converged V_v : {new_V_v}')

    list_grads = np.array(list(map(lambda x: mp._model.wind.d_value(x), sample_sphere(x_init, T * (v_a + V_v)))))
    N_v = np.max(list_grads)
    print(f'N_v : {N_v}')
    print(f'V_v : {V_v}')
    print(f'Trivial lb : {1. / (v_a + V_v)}')

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
    """
    new_points = enlarge(points, T ** 2 / 2. * (v_a + V_v) * N_v)
    mp.trajs.append(
        Trajectory(np.zeros(len(new_points)), new_points, np.zeros(len(new_points)), len(new_points), type="approx"))
    """
    """
    fast_approx = np.zeros((N_points, 2))
    for k, s in enumerate(unit_sphere):
        fast_approx[k, :] = x_init + T * (v_a + V_v) * s
    mp.trajs.append(Trajectory(np.zeros(len(fast_approx)), fast_approx, np.zeros(len(fast_approx)), len(fast_approx),
                               type=TRAJ_INT))
    """

    grad_vw = mp._model.wind.d_value(x_init)
    print(f'grad_vw :\n{grad_vw}')
    vps, eigvecs = np.linalg.eig(grad_vw)

    def grad_norm(x):
        wind = mp._model.wind.value(x)
        return mp._model.wind.d_value(x).transpose().dot(wind / np.linalg.norm(wind))

    print(f'norm grad norm : {np.linalg.norm(grad_norm(x_init))}')
    print(f'eigvecs :\n{eigvecs}')
    print(f'vps :\n{vps}')
    e1 = eigvecs[:, 0]
    # mp.display.map_adjoint.arrow(0., 0., e1[0], e1[1])
    e = mp._model.wind.value(x_init) / np.linalg.norm(mp._model.wind.value(x_init))
    # mp.display.map_adjoint.arrow(0., 0., e[0], e[1], color='blue')
    e = grad_vw.dot(e)
    # mp.display.map_adjoint.arrow(0., 0., e[0], e[1], color='red')
    X, Y = np.meshgrid(np.linspace(-0.05, 1.05, 50), np.linspace(-1.05, 1.05, 50))
    Z = np.array([[np.linalg.norm(grad_norm(np.array([X[i, j], Y[i, j]]))) for j in range(X.shape[1])] for i in range(X.shape[0])])
    mp.display.map.contour(X, Y, Z, levels=np.linspace(0., N_shooting_iter/T, 100))

    cartesian = np.dstack((X, Y)).reshape(-1, 2)

    def e_minus(x):
        d_wind = mp._model.wind.d_value(x)
        dw_x = d_wind[0, 0]
        dw_y = d_wind[0, 1]
        lambda_0 = np.sqrt(dw_x ** 2 + dw_y ** 2)
        return np.array([- dw_y, lambda_0 + dw_x])

    uv = np.array(list(map(mp._model.wind.e_minus, list(cartesian))))
    U, V = uv[:, 0], uv[:, 1]
    norms = np.sqrt(U ** 2 + V ** 2)

    mp.display.map.quiver(X, Y, U / norms, V / norms, cmap='viridis')

    # Clover shape

    # accel = np.zeros((N_points, 2))
    # for k, s in enumerate(unit_sphere):
    #     accel[k, :] = np.einsum('i,ij,j', s, grad_vw, s) * s
    # mp.trajs.append(AugmentedTraj(np.zeros(N_points), np.zeros((N_points, 2)), accel, np.zeros(N_points), N_points))
    #
    # accel = np.zeros((N_points, 2))
    # for k, s in enumerate(unit_sphere):
    #     accel[k, :] = - grad_vw.dot(s)
    # mp.trajs.append(AugmentedTraj(np.zeros(N_points), np.zeros((N_points, 2)), accel, np.zeros(N_points), N_points, type='pmp'))

    accel = np.zeros((N_points, 2))
    for k, s in enumerate(unit_sphere):
        accel[k, :] = - grad_vw.dot(s) + np.einsum('i,ij,j', s, grad_vw, s) * s
    mp.trajs.append(
        AugmentedTraj(np.zeros(N_points), np.zeros((N_points, 2)), accel, np.zeros(N_points), N_points, type='pmp'))

    # Get time-optimal candidate trajectories as integrals of
    # the augmented system using the shooting method
    # The control law is a result of the integration and is
    # thus implicitly defined
    for p in list_p:
        shoot = Shooting(zermelo_model.dyn, x_init, T, N_iter=N_shooting_iter)
        shoot.set_adjoint(p)
        aug_traj = shoot.integrate()
        aug_traj.adjoints *= -1.
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
    mp.plot_trajs(color_mode="reachability", selection=False)


if __name__ == '__main__':
    example1()
