import numpy as np

from src.mermoz import MermozProblem
from src.shooting import Shooting
from src.trajectory import Trajectory


class Solver:
    """
    Solve the Mermoz problem
    """

    def __init__(self, mp: MermozProblem, x_init, T, N_disc_init=20):
        self.mp = mp
        self.x_init = np.zeros(2)
        self.x_init[:] = x_init
        self.T = T
        self.N_disc_init = N_disc_init

        self.N_opti = 1

        self.dirs = np.linspace(-np.pi / 2., np.pi / 2, self.N_disc_init)
        self.neighbours = {}
        for i in range(self.N_disc_init):
            self.neighbours[i] = []
            if i - 1 >= 0:
                self.neighbours[i].append(i - 1)
            if i + 1 < self.N_disc_init:
                self.neighbours[i].append(i + 1)
        self.distances = {}
        for i in self.neighbours.keys():
            for j in self.neighbours[i]:
                if sorted((i, j)) not in self.distances:
                    self.distances[i, j] = np.infty
        self.to_shoot = [True for _ in range(self.N_disc_init)]
        self.trajs = []
        self.opti_test = []
        self.opti_indexes = []

    def add_neighbour(self, k, i, j):
        self.neighbours[i].append(k)
        self.neighbours[j].append(k)
        self.neighbours[k] = [i, j]
        self.distances[sorted((k, i))] = np.infty
        self.distances[sorted((k, j))] = np.infty

    def is_opti(self, traj: Trajectory):
        b = False
        for k in range(traj.last_index):
            if np.linalg.norm(traj.points[k] - np.array([self.mp._model.x_f, 0.])) < 1e-1:
                b = True
                break
        return b

    def solve(self):

        quit = False
        while True:
            # Shooting phase
            for k, d in enumerate(self.dirs):
                if self.to_shoot[k]:
                    shoot = Shooting(self.mp._model.dyn, self.x_init, self.T, adapt_ts=True, N_iter=1000,
                                     domain=self.mp.domain, stop_on_failure=True)
                    shoot.set_adjoint(d)
                    self.trajs.append(shoot.integrate())
                    self.to_shoot[k] = False
                    self.opti_test.append(False)

            # Check optimality
            for k, d in enumerate(self.dirs):
                if not self.opti_test[k]:
                    self.opti_test[k] = True
                    b = self.is_opti(self.trajs[k])
                    if b:
                        self.opti_indexes.append(k)
                        if len(self.opti_indexes) >= self.N_opti:
                            quit = True
            if quit:
                break

            # Compute distances
            last_point_1 = np.zeros(2)
            last_point_2 = np.zeros(2)
            for k, traj in enumerate(self.trajs):
                last_point_2[:] = traj.points[traj.last_index - 1]
                if k > 0:
                    self.distances[k - 1] = np.linalg.norm(last_point_2 - last_point_1)
                last_point_1[:] = last_point_2
