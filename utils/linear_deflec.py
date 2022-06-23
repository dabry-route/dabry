from mermoz.problem import IndexedProblem
from mermoz.misc import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Compute the deflection from geodesic of the quickest path in a linear wind
    pb = IndexedProblem(1)

    scale = 0.01
    pb.model.v_a = 1.
    pb.x_init[:] = np.zeros(2)
    pb.x_target[:] = np.array((1, 0.))

    N = 100
    scales = np.linspace(0.001, 10., N)
    y_max = np.zeros(N)
    v_max = np.zeros(N)
    fig, ax = plt.subplots()
    ax.axis('equal')
    for k, scale in enumerate(scales):
        gradient = scale * pb.model.v_a / (pb.x_target[0] - pb.x_init[0])
        alyt_traj = linear_wind_alyt_traj(pb.model.v_a, gradient, pb.x_init, pb.x_target)
        ax.scatter(alyt_traj.points[:, 0], alyt_traj.points[:, 1], s=1.)
        # plt.scatter(alyt_traj.points[:, 0], alyt_traj.points[:, 1])
        y_max[k] = alyt_traj.points[:, 1].max()
        v_max[k] = y_max[k] * scale

    fig, ax = plt.subplots()
    ax.plot(scales, y_max)
    #ax.plot(scales, v_max)

    plt.show()
