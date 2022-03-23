import math
import os
import time
import numpy as np
from matplotlib import pyplot as plt

from .problem import MermozProblem
from .shooting import Shooting
from .trajectory import Trajectory


class Solver:
    """
    Solve the Mermoz problem
    """

    def __init__(self,
                 mp: MermozProblem,
                 x_init,
                 x_target,
                 T,
                 min_init_angle,
                 max_init_angle,
                 N_disc_init=20,
                 neighb_ceil=1e-1,
                 int_prec_factor=1e-2,
                 opti_ceil=5e-2,
                 n_min_opti=1,
                 adaptive_int_step=True,
                 N_iter=1000):
        self.mp = mp
        self.x_init = np.zeros(2)
        self.x_init[:] = x_init
        self.x_target = np.zeros(2)
        self.x_target[:] = x_target
        self.T = T
        self.N_disc_init = N_disc_init
        self.neighb_ceil = neighb_ceil
        self.int_prec_factor = int_prec_factor
        self.adaptive_int_step = adaptive_int_step
        self.N_iter = N_iter
        self.opti_ceil = opti_ceil

        self.output_dir = os.path.join(os.getcwd(), '../..', 'output', '.tmp')

        self.min_init_angle = min_init_angle
        self.max_init_angle = max_init_angle
        self.n_min_opti = n_min_opti
        dirs = np.linspace(self.min_init_angle, self.max_init_angle, self.N_disc_init)
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
        self.distances_queue = []
        for i in self.neighbours.keys():
            for j in self.neighbours[i]:
                if j > i:
                    self.distances_queue.append((i, j))
        self.to_shoot = [True for _ in range(self.N_disc_init)]
        self.trajs = {}
        self.last_points = {}
        self.parents = {}
        self.opti_test = []
        self.opti_indexes = []
        self.opti_time = {}

    def setup(self):
        for f in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, f))
        self.log_config()

    def add_neighbour(self, k, i, j):
        self.neighbours[i].append(k)
        self.neighbours[j].append(k)
        self.neighbours[k] = [i, j]
        t = (k, i) if i > k else (i, k)
        self.distances_queue.append(t)
        t = (k, j) if j > k else (j, k)
        self.distances_queue.append(t)

    def is_opti(self, traj: Trajectory):
        b = False
        k = -1
        for k in range(traj.last_index):
            if np.linalg.norm(traj.points[k] - self.x_target) < self.opti_ceil:
                b = True
                break
        return b, k

    def config_to_string(self):
        ln = {
            'problem': 'Problem',
            'time': 'Problem time horizon',
            'x_init': 'Problem starting point',
            'init_shoot_angles': 'Init. shoot angles (min, max, nb)',
            'mode': 'Mode',
            'steps': 'Max. steps' if self.adaptive_int_step else 'Steps',
            'neighb_ceil': 'Neighbour distance ceil',
            'int_prec_factor' : 'Integration precision factor'
        }
        padding = max(map(len, ln.values()))

        def format_line(name, suffix, column=True):
            if not isinstance(suffix, str):
                suffix = str(suffix)
            return ('{:>' + str(padding) + '}').format(name) + (" : " if column else "   ") + suffix + '\n'

        s = format_line('', '** SOLVER CONFIG **', column=False)
        s += format_line(ln['problem'], self.mp)
        s += format_line(ln['time'], f'{self.T:.3f} s')
        s += format_line(ln['x_init'], f'({self.x_init[0]:.3f}, {self.x_init[1]:.3f})')
        s += format_line(ln['init_shoot_angles'],
                         f'{self.min_init_angle:.3f}, {self.max_init_angle:.3f}, {self.N_disc_init}')
        s += format_line(ln['mode'],
                         'Adaptive integration step' if self.adaptive_int_step else 'Fixed integration step')
        s += format_line(ln['steps'], self.N_iter)
        s += format_line(ln['neighb_ceil'], f'{self.neighb_ceil:.2e}')
        s += format_line(ln['int_prec_factor'], f'{self.int_prec_factor:.2e}')
        return s

    def log_config(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            f.writelines(self.config_to_string())

    def solve(self):

        quit = False
        t_start = time.time()
        d_proc = 0.
        d_post_proc = 0.
        iter = 0
        while True:
            # Shooting phase
            t_start2 = time.time()
            count = 0
            for ts in self.to_shoot:
                if ts:
                    count += 1
            print(f'* Iteration {iter} - {count} trajectories - {len(self.opti_indexes)}/{self.n_min_opti} optima')
            print(f"    * Shooting phase... ", end='')
            int_steps = []
            for k, d in self.dirs.items():
                if self.to_shoot[k]:
                    shoot = Shooting(self.mp._model.dyn, self.x_init, self.T, adapt_ts=self.adaptive_int_step,
                                     N_iter=self.N_iter, domain=self.mp.domain, fail_on_maxiter=True,
                                     factor=self.int_prec_factor)
                    shoot.set_adjoint(np.array([-np.cos(d), -np.sin(d)]))
                    traj = shoot.integrate()
                    self.trajs[k] = traj
                    int_steps.append(traj.last_index)
                    vect = np.zeros(2)
                    vect[:] = traj.points[traj.last_index - 1]
                    self.last_points[k] = vect
                    self.to_shoot[k] = False
                    self.opti_test.append(False)
            t_end2 = time.time()
            duration = t_end2 - t_start2
            d_proc += duration
            print(
                f"Done ({duration:.3f} s, {(t_end2 - t_start2) / count * 1000:.1f} ms/traj, {math.floor(sum(int_steps) / len(int_steps))} mean int. steps)")
            # Check optimality
            t_start2 = time.time()
            print(f"    * Optimality check... ", end='')
            prev_len = len(self.opti_indexes)
            for k, d in enumerate(self.dirs):
                if not self.opti_test[k]:
                    self.opti_test[k] = True
                    b, index = self.is_opti(self.trajs[k])
                    if b:
                        self.opti_indexes.append(k)
                        self.trajs[k].type = 'optimal'
                        self.opti_time[k] = self.trajs[k].timestamps[index]
                        if len(self.opti_indexes) >= self.n_min_opti:
                            quit = True
            t_end2 = time.time()
            duration = t_end2 - t_start2
            d_proc += duration
            print(f"{len(self.opti_indexes) - prev_len} found ({duration:.3f} s)")
            if quit:
                print(f"* {len(self.opti_indexes)}/{self.n_min_opti} optima found, exiting")
                break

            # Compute distances
            t_start2 = time.time()
            print(f"    * Computing endpoint distances... ", end='')
            for t in self.distances_queue:
                self.distances[t] = np.linalg.norm(self.last_points[t[0]] - self.last_points[t[1]])
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
                        if not parents and self.distances[i, j] >= self.neighb_ceil:
                            self.dirs[len(self.dirs)] = (self.dirs[i] + self.dirs[j]) / 2
                            k = len(self.to_shoot)
                            self.to_shoot.append(True)
                            self.parents[i, j] = True
                            next_neighbours.append((k, i, j))
            for nn in next_neighbours:
                self.add_neighbour(nn[0], nn[1], nn[2])
            if len(next_neighbours) == 0:
                break
            t_end2 = time.time()
            duration = t_end2 - t_start2
            d_proc += duration
            print(f"Done ({duration:.3f} s)")
            iter += 1
            t_start2 = time.time()
            print(f'    * Plotting results...', end='')
            self.mp.trajs = self.trajs.values()
            self.mp.plot_trajs(color_mode="reachability")
            plt.savefig(os.path.join(self.output_dir, f'solver_{iter}.png'), dpi=300)
            t_end2 = time.time()
            duration = t_end2 - t_start2
            d_post_proc += duration
            print(f"Done ({duration:.3f} s)")
        t_end = time.time()
        print('')
        print(f'* End of loop')
        print(f'    * Processing      :  {d_proc:.3f} s')
        print(f'    * Post-processing :  {d_post_proc:.3f} s')
        print(f'    * Total           :  {t_end - t_start:.3f} s')
        if len(self.opti_indexes) == 0:
            print("No optimum found")
        else:
            for k in self.opti_indexes:
                print(k)
                print(self.opti_time[k])
                self.trajs[k].type = 'optimal'
            self.mp.trajs = self.trajs.values()
            self.mp.plot_trajs(color_mode="reachability")
            iter += 1
            plt.savefig(os.path.join(self.output_dir, f'solver_{iter}.png'), dpi=300)