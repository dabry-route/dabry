import numpy as np

from dabry.feedback import FunFB
from dabry.misc import Utils
from dabry.problem import NavigationProblem
from dabry.rft import RFT
from dabry.solver_ef import EFOptRes, Particle
from dabry.stoppingcond import DistanceSC

"""
solver_rp.py
Solver for navigation problems using reachability front tracking.

Copyright (C) 2021 Bastien Schnitzler 
(bastien dot schnitzler at live dot fr)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


class SolverRP:
    """
    Minimum time trajectory planning solver
    using rough Reachability Front Tracking (RFT)
    """

    def __init__(self,
                 mp: NavigationProblem,
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

        self.geod_l = Utils.distance(self.x_init, self.x_target, coords=self.mp.coords)

        if max_time is None:
            self.T = 1.5 * self.geod_l / mp.model.v_a
        else:
            self.T = max_time

        self.opti_ceil = self.geod_l / 50
        self.neighb_ceil = self.opti_ceil / 2.

        self.t_target = None

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
            self.t_target = self.rft.get_time(self.mp.x_target)
            duration = self.t_target - self.mp.model.wind.t_start
            traj = self.get_opti_traj()
            fake_pcl = Particle(0, 0, self.t_target, traj.points[-1], np.zeros(2), duration)
            bests = {0: fake_pcl}
            trajs = {0: traj}
        else:
            bests = {}
            trajs = {}
        return EFOptRes(status, bests, trajs, self.mp)

    def get_opti_traj(self, int_step=100):
        self.mp.load_feedback(FunFB(lambda x: self.control(x, backward=True), no_time=True))
        sc = DistanceSC(lambda x: self.mp.distance(x, self.mp.x_init), self.mp.geod_l * 0.001)
        duration = self.t_target - self.mp.model.wind.t_start
        traj = self.mp.integrate_trajectory(self.mp.x_target, sc, int_step=duration / int_step,
                                            t_init=self.t_target,
                                            max_iter=100,
                                            backward=True)
        traj.timestamps = traj.timestamps[:traj.last_index + 1]
        traj.points = traj.points[:traj.last_index + 1]
        traj.controls = traj.controls[:traj.last_index + 1]
        traj.flip()
        traj.info = 'rft'
        traj.type = Utils.TRAJ_INT
        return traj
