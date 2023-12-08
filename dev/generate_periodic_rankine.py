import numpy as np

from dabry.ddf_manager import DDFmanager
from dabry.misc import Utils
from dabry.wind import RankineVortexWind, DiscreteWind

v_a = 1
coords = Utils.COORD_CARTESIAN

bl = np.array((-0.5, -0.5))
tr = np.array((0.5, 0.5))

x_init = np.array([0., 0.])
x_target = np.array([1., 0.])
nt = 20
xs = np.linspace(0., 0., nt)
ys = np.linspace(-0.5, 0.5, nt)

xs_up = np.array(xs)
ys_up = np.linspace(0.5, 1.5, nt)

xs_dw = np.array(xs)
ys_dw = np.linspace(-1.5, -0.5, nt)

omega = np.stack((xs, ys), axis=1)
omega_up = np.stack((xs_up, ys_up), axis=1)
omega_dw = np.stack((xs_dw, ys_dw), axis=1)
gamma = -1. * np.ones(nt)
radius = 1e-1 * np.ones(nt)

vortex = RankineVortexWind(omega, gamma, radius, t_end=2)
vortex_up = RankineVortexWind(omega_up, gamma, radius, t_end=2)
vortex_dw = RankineVortexWind(omega_dw, gamma, radius, t_end=2)

wind = vortex + vortex_up + vortex_dw

disc = DiscreteWind()
disc.load_from_wind(vortex, 51, 51, bl, tr, coords, nt=nt)
ddf = DDFmanager()
ddf.setup()
ddf.set_case('periodic_rankine')
ddf.dump_wind(disc)