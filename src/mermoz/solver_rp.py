import time
from math import atan2

from mermoz.model import ZermeloGeneralModel
from mermoz.problem import MermozProblem
from mermoz.rft import RFT
from mermoz.misc import *
from mermoz.solver import Solver
from mermoz.wind import DiscreteWind


class SolverRP:
    """
    Minimum time trajectory planning solver
    using rough Reachability Front Tracking (RFT)
    to guess arrival time and arrival heading at destination
    and then Pontryagin's Maximum Principle to cheaply shoot extremal
    trajectories on previous parameters
    """

    def __init__(self,
                 mp: MermozProblem,
                 nx_rft,
                 ny_rft,
                 nt_rft,
                 nt_pmp=1000,
                 extremals=True,
                 only_opti_extremals=False):
        self.mp = mp
        self.x_init = np.zeros(2)
        self.x_init[:] = mp.x_init
        self.x_target = np.zeros(2)
        self.x_target[:] = mp.x_target
        self.nx_rft = nx_rft
        self.ny_rft = ny_rft
        self.nt_rft = nt_rft
        # Effective number of steps to reach target
        # Means nt_rft_eff is the smallest n such that
        # phi[n-1](x_target) > 0. and phi[n](x_target) < 0.
        self.nt_rft_eff = None

        self.extremals = extremals
        self.only_opti_extremals = only_opti_extremals

        self.nt_pmp = nt_pmp

        self.geod_l = distance(self.x_init, self.x_target, coords=self.mp.coords)

        self.T = 2. * self.geod_l / mp.model.v_a

        self.opti_ceil = self.geod_l / 50
        self.neighb_ceil = self.opti_ceil / 2.

        l = 2. * self.geod_l
        bl_rft = np.zeros(2)
        tr_rft = np.zeros(2)
        bl_rft[:] = mp.bl # (self.x_init + self.x_target) / 2. - np.array((l / 2., l / 2.))
        tr_rft[:] = mp.tr # (self.x_init + self.x_target) / 2. + np.array((l / 2., l / 2.))

        self.rft = RFT(bl_rft, tr_rft, self.T, nx_rft, ny_rft, nt_rft, mp, self.x_init, kernel='matlab',
                       coords=mp.coords)

        # Setting the reverse-time Mermoz navigation problem
        zermelo_model = ZermeloGeneralModel(-self.mp.model.v_a, coords=self.mp.coords)
        if type(self.mp.model.wind) == DiscreteWind:
            wind = DiscreteWind()
            bl = (self.mp.model.wind.x_min, self.mp.model.wind.y_min)
            tr = (self.mp.model.wind.x_max, self.mp.model.wind.y_max)
            wind.load_from_wind(-1. * self.mp.model.wind, self.mp.model.wind.nx, self.mp.model.wind.ny,
                                bl, tr, self.mp.coords)
        else:
            wind = -1. * self.mp.model.wind
        zermelo_model.update_wind(wind)
        self.mp_pmp = MermozProblem(zermelo_model, self.x_init, self.x_target, self.mp.coords, autodomain=False,
                                    domain=self.mp.domain, mask_land=False)

    def solve(self):
        print(f"Tracking reachability front ({self.nx_rft}x{self.ny_rft}x{self.nt_rft})... ")
        t_start = time.time()
        self.rft.compute()

        t_end = time.time()
        time_rft = t_end - t_start
        self.mp.trajs.append(self.rft.backward_traj(self.mp.x_target, self.mp.x_init, self.opti_ceil,
                                                    self.T, self.mp.model, N_disc=1000))
        print(f"Done ({time_rft:.3f} s)")
        time_pmp = 0.
        normal, T, self.nt_rft_eff = self.rft.get_normal(self.x_target, compute=False)
        if self.extremals:
            self.T = 1.1 * T

            auto_psi = atan2(normal[1], normal[0])

            psi_min = auto_psi - DEG_TO_RAD * 10.
            psi_max = auto_psi + DEG_TO_RAD * 10.

            solver = Solver(self.mp_pmp,
                            self.x_target,
                            self.x_init,
                            T,
                            psi_min,
                            psi_max,
                            '/home/bastien/Documents/work',
                            N_disc_init=10,
                            opti_ceil=self.opti_ceil,
                            neighb_ceil=self.neighb_ceil,
                            n_min_opti=1,
                            adaptive_int_step=False,
                            nt_pmp=self.nt_pmp)

            solver.log_config()

            solver.setup()

            t_start = time.time()
            success = solver.solve_fancy(dump_only_opti=self.only_opti_extremals)
            if not success:
                print(f"Shooting failed, enlarging sector")
                for sector in [(psi_min - 10. * DEG_TO_RAD, psi_min),
                               (psi_max, psi_max + 10. * DEG_TO_RAD),
                               (psi_min - 20. * DEG_TO_RAD, psi_min - 10. * DEG_TO_RAD),
                               (psi_max + 10. * DEG_TO_RAD, psi_max + 20. * DEG_TO_RAD)]:
                    solver = Solver(self.mp_pmp,
                                    self.x_target,
                                    self.x_init,
                                    T,
                                    sector[0],
                                    sector[1],
                                    '/home/bastien/Documents/work',
                                    N_disc_init=10,
                                    opti_ceil=self.opti_ceil,
                                    neighb_ceil=self.neighb_ceil,
                                    n_min_opti=1,
                                    adaptive_int_step=False,
                                    nt_pmp=self.nt_pmp)
                    success = solver.solve_fancy()
                    if success:
                        break

            t_end = time.time()
            time_pmp = t_end - t_start
            print(f"Done ({time_pmp:.3f} s)")

        print(f'Total RP solver time : {time_rft + time_pmp:.3f} s')
