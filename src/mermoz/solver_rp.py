from mermoz.problem import MermozProblem
from mermoz.rft import RFT
from mermoz.misc import *
from mermoz.feedback import FunFB
from mermoz.solver_ef import EFOptRes
from mermoz.stoppingcond import DistanceSC


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
                 max_time=None):
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

        self.nt_pmp = nt_pmp

        self.geod_l = distance(self.x_init, self.x_target, coords=self.mp.coords)

        if max_time is None:
            self.T = 1.5 * self.geod_l / mp.model.v_a
        else:
            self.T = max_time

        self.opti_ceil = self.geod_l / 50
        self.neighb_ceil = self.opti_ceil / 2.

        self.reach_time = None

        bl_rft = np.zeros(2)
        tr_rft = np.zeros(2)
        bl_rft[:] = mp.bl  # (self.x_init + self.x_target) / 2. - np.array((l / 2., l / 2.))
        tr_rft[:] = mp.tr  # (self.x_init + self.x_target) / 2. + np.array((l / 2., l / 2.))

        self.rft = RFT(bl_rft, tr_rft, self.T, nx_rft, ny_rft, nt_rft, mp, kernel='matlab')
        self.rft.setup_matlab()

    def control(self, x, backward=False):
        return self.rft.control(x, backward)

    def solve(self):
        self.rft.compute()
        status = 2 if self.rft.has_reached() else 0
        if status:
            self.reach_time = self.rft.get_time(self.mp.x_target) - self.mp.model.wind.t_start
            traj = self.get_opti_traj()
            bests = {
                0: {'cost': self.reach_time,
                    'duration': self.reach_time,
                    'traj': traj,
                    'adjoint': None}
            }
        else:
            bests = {}
        return EFOptRes(status, bests)

    def get_opti_traj(self, int_step=100):
        self.mp.load_feedback(FunFB(lambda x: self.control(x, backward=True), no_time=True))
        sc = DistanceSC(lambda x: self.mp.distance(x, self.mp.x_init), self.mp.geod_l * 0.001)
        traj = self.mp.integrate_trajectory(self.mp.x_target, sc, int_step=self.reach_time / int_step,
                                            t_init=self.mp.model.wind.t_start + self.reach_time,
                                            backward=True)
        traj.timestamps = traj.timestamps[:traj.last_index + 1]
        traj.points = traj.points[:traj.last_index + 1]
        traj.controls = traj.controls[:traj.last_index + 1]
        traj.flip()
        traj.info = 'rft'
        traj.type = TRAJ_INT
        return traj
