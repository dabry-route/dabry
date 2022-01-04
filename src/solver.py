import time

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from src.mermoz import MermozProblem
from src.shooting import Shooting
from src.trajectory import Trajectory


class Solver:
    """
    Solve the Mermoz problem
    """

    def __init__(self, mp: MermozProblem, x_init, T, N_disc_init=20, ceil=1e-1, n_min_opti=1):
        self.mp = mp
        self.x_init = np.zeros(2)
        self.x_init[:] = x_init
        self.T = T
        self.N_disc_init = N_disc_init
        self.ceil = ceil

        self.n_min_opti = n_min_opti
        dirs = np.linspace(-3 * np.pi / 8., 3 * np.pi / 8, self.N_disc_init)
        self.dirs = {}
        for k, d in enumerate(dirs):
            self.dirs[k] = d
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
                if j > i:
                    self.distances[i, j] = np.infty
        self.to_shoot = [True for _ in range(self.N_disc_init)]
        self.trajs = {}
        self.last_points = {}
        self.parents = {}
        self.opti_test = []
        self.opti_indexes = []
        self.opti_time = {}

    def add_neighbour(self, k, i, j):
        self.neighbours[i].append(k)
        self.neighbours[j].append(k)
        self.neighbours[k] = [i, j]
        t = (k, i) if i > k else (i, k)
        self.distances[t] = np.infty
        t = (k, j) if j > k else (j, k)
        self.distances[t] = np.infty

    def is_opti(self, traj: Trajectory):
        b = False
        for k in range(traj.last_index):
            if np.linalg.norm(traj.points[k] - np.array([self.mp._model.x_f, 0.])) < 5e-2:
                b = True
                break
        return b, k

    def solve(self):

        quit = False
        t_start = time.time()
        iter = 0
        while True:
            # Shooting phase
            for k, d in self.dirs.items():
                if self.to_shoot[k]:
                    print(f"Shooting {d}")
                    shoot = Shooting(self.mp._model.dyn, self.x_init, self.T, adapt_ts=True, N_iter=10000,
                                     domain=self.mp.domain, stop_on_failure=True, ceil=1e-2)
                    shoot.set_adjoint(np.array([-np.cos(d), -np.sin(d)]))
                    traj = shoot.integrate()
                    self.trajs[k] = traj
                    vect = np.zeros(2)
                    vect[:] = traj.points[traj.last_index - 1]
                    self.last_points[k] = vect
                    self.to_shoot[k] = False
                    self.opti_test.append(False)

            # Check optimality
            for k, d in enumerate(self.dirs):
                if not self.opti_test[k]:
                    self.opti_test[k] = True
                    b, index = self.is_opti(self.trajs[k])
                    if b:
                        self.opti_indexes.append(k)
                        self.opti_time[k] = self.trajs[k].timestamps[index]
                        if len(self.opti_indexes) >= self.n_min_opti:
                            quit = True
            if quit:
                break

            # Compute distances
            for i in self.neighbours.keys():
                for j in self.neighbours[i]:
                    if j > i:
                        self.distances[i, j] = np.linalg.norm(self.last_points[i] - self.last_points[j])
            next_neighbours = []
            # Check for points to add
            for i in self.neighbours.keys():
                for j in self.neighbours[i]:
                    if j > i:
                        parents = False
                        try:
                            parents = self.parents[i, j]
                        except KeyError:
                            pass
                        if not parents and self.distances[i, j] >= self.ceil:
                            self.dirs[len(self.dirs)] = (self.dirs[i] + self.dirs[j]) / 2
                            k = len(self.to_shoot)
                            self.to_shoot.append(True)
                            self.parents[i, j] = True
                            next_neighbours.append((k, i, j))
            for nn in next_neighbours:
                self.add_neighbour(nn[0], nn[1], nn[2])
            if len(next_neighbours) == 0:
                break
            print(self.distances)
            # if iter == 10:
            #     self.mp.trajs = self.trajs.values()
            #     self.mp.plot_trajs()
            #     plt.show()
            iter += 1
            pass
        t_end = time.time()
        print(f'End of loop ({t_end - t_start:.3f} s)')
        if len(self.opti_indexes) == 0:
            print("No optimum found")
        else:
            for k in self.opti_indexes:
                print(k)
                print(self.opti_time[k])
                self.trajs[k].type = 'optimal'
            self.mp.trajs = self.trajs.values()
            self.mp.plot_trajs(color_mode="reachability")
            plt.show()
