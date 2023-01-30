from dabry.misc import Utils
import pyproj
import numpy as np

if __name__ == '__main__':
    center = np.array((20., 30.))
    proj = pyproj.Proj(proj='ortho', lon_0=center[0], lat_0=center[1])
    points = np.column_stack((center[0] + 60. * (np.random.rand(100) - 0.5),
                              (center[1] + 60. * (np.random.rand(100) - 0.5))))
    perso = np.array(list(map(
        lambda x: Utils.proj_ortho(x[0], x[1], Utils.DEG_TO_RAD * center[0], Utils.DEG_TO_RAD * center[1]), Utils.DEG_TO_RAD * points)))
    package = np.array(proj(points[:, 0], points[:, 1])).transpose()

    # Test analytical projection is consistent with package projection
    # This one shall be negligible compared to projected coordinates order of magnitude (typically 1e5)
    print('Test proj_ortho')
    print(f'{np.sum(np.abs(perso - package))} (negligible compared to {np.linalg.norm(package)} ?)')

    perso_inv = np.array(list(map(
        lambda x: Utils.proj_ortho_inv(x[0], x[1], Utils.DEG_TO_RAD * center[0], Utils.DEG_TO_RAD * center[1]), perso)))

    print('Test proj_ortho_inv')
    print(
        f'{np.sum(np.abs(Utils.DEG_TO_RAD * points - perso_inv))} (negligible compared to {np.linalg.norm(Utils.DEG_TO_RAD * points)} ?)')


    def m_diff(f, x, dx=1e-8):
        n = x.shape[0]
        vdx = np.diag(n * (1,)) * dx
        return np.column_stack(((f(x + vdx[i]) - f(x)) / vdx[i, i] for i in range(n)))


    d_manual = np.array(list(map(
        lambda x: m_diff((
            lambda y: Utils.proj_ortho(y[0], y[1], Utils.DEG_TO_RAD * center[0], Utils.DEG_TO_RAD * center[1])),
            x),
        Utils.DEG_TO_RAD * points)))

    d_analytical = np.array(list(map(
        lambda x: Utils.d_proj_ortho(x[0], x[1], Utils.DEG_TO_RAD * center[0], Utils.DEG_TO_RAD * center[1]), Utils.DEG_TO_RAD * points)))

    print('Test d_proj_ortho')
    print(f'{np.sum(np.abs(d_manual - d_analytical))} (negligible compared to {np.linalg.norm(d_manual)} ?)')

    d_manual_inv = np.array(list(map(
        lambda x: m_diff((
            lambda y: Utils.proj_ortho_inv(y[0], y[1], Utils.DEG_TO_RAD * center[0], Utils.DEG_TO_RAD * center[1])),
            x, dx=1e-2),
        package)))

    d_analytical_inv = np.array(list(map(
        lambda x: Utils.d_proj_ortho_inv(x[0], x[1], Utils.DEG_TO_RAD * center[0], Utils.DEG_TO_RAD * center[1]), package)))

    print('Test d_proj_ortho_inv')
    print(d_analytical_inv[0])
    print(d_manual_inv[0])
    print(
        f'{np.sum(np.abs(d_manual_inv - d_analytical_inv))} (negligible compared to {np.linalg.norm(d_manual_inv)} ?)')
