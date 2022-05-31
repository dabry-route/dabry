import math
import os
import time
import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from heapq import heappush, heappop

from mermoz.problem import MermozProblem
from mermoz.shooting import Shooting
from mermoz.trajectory import Trajectory
from mermoz.misc import *


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
                 output_dir,
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

        self.output_dir = output_dir

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
        # Values in queue are in format
        # (Direction to shoot, index, lower parent direction, its index, its depth,
        # upper parent direction, its index, its depth)
        self.shoot_queue = []

        self.trajs = {}
        self.last_points = {}
        self.parents = {}
        self.opti_test = []
        self.opti_test_dict = {}
        self.opti_indexes = []
        self.opti_headings = {}
        self.opti_time = {}

        self.iter = 0

    def setup(self):
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.log_config()

    def add_neighbour(self, k, i, j):
        self.neighbours[i].append(k)
        self.neighbours[j].append(k)
        self.neighbours[k] = [i, j]
        t = (k, i) if i > k else (i, k)
        self.distances_queue.append(t)
        t = (k, j) if j > k else (j, k)
        self.distances_queue.append(t)

    def distance(self, x1, x2):
        if self.mp.coords == COORD_GCS:
            # x1, x2 shall be vectors (lon, lat) in radians
            return geodesic_distance(x1, x2)
        else:
            # x1, x2 shall be cartesian vectors in meters
            return np.linalg.norm(x1 - x2)

    def is_opti(self, traj: Trajectory):
        b = False
        k = -1
        for k in range(traj.last_index - 1, -1, -1):
            if self.distance(traj.points[k], self.x_target) < self.opti_ceil:
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
            'int_prec_factor': 'Integration precision factor'
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

    def solve(self, plotting=False):

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
                    shoot = Shooting(self.mp.model.dyn, self.x_init, self.T, adapt_ts=self.adaptive_int_step,
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
            self.mp.trajs = list(self.trajs.values())
            if plotting:
                print(f'    * Plotting results...', end='')
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
            self.mp.trajs = list(self.trajs.values())
            if plotting:
                self.mp.plot_trajs(color_mode="reachability")
                iter += 1
                plt.savefig(os.path.join(self.output_dir, f'solver_{iter}.png'), dpi=300)

    def print_iteration(self, n_trajs):
        print(f'* Iteration {self.iter} - {n_trajs} trajectories - {len(self.opti_indexes)}/{self.n_min_opti} optima')

    def adj_from_dir(self, d):
        if self.mp.coords == COORD_CARTESIAN:
            return np.array([-np.cos(d), -np.sin(d)])
        elif self.mp.coords == COORD_GCS:
            return np.array([-np.sin(d), -np.cos(d)])

    def solve_fancy(self, max_depth=10, debug=False):

        t_start = time.time()
        self.iter = 0
        int_steps = []
        k_max = len(self.dirs.keys())
        dirs_list = []
        dirs_queue = []
        for k, d in self.dirs.items():
            print(f'Shooting {d * 180 / pi:.1f}')
            dirs_list.append(d)
            shoot = Shooting(self.mp.model.dyn, self.x_init, self.T, adapt_ts=self.adaptive_int_step,
                             N_iter=self.N_iter, domain=self.mp.domain, fail_on_maxiter=True,
                             factor=self.int_prec_factor, coords=self.mp.coords)
            shoot.set_adjoint(self.adj_from_dir(d))
            traj = shoot.integrate()
            self.trajs[k] = traj
            lp = np.zeros(2)
            lp[:] = traj.points[traj.last_index - 1]
            self.last_points[k] = lp
            int_steps.append(traj.last_index)
            if k != 0:
                d1 = self.dirs[k - 1]
                d2 = self.dirs[k]
                p1 = min(
                    self.distance(p, self.x_target) for p in self.trajs[k - 1].points[:self.trajs[k - 1].last_index])
                p2 = min(self.distance(p, self.x_target) for p in self.trajs[k].points[:self.trajs[k].last_index])
                l1 = 0.5  # p1 / (p1 + p2)
                l2 = 0.5  # 1 - l1
                new_dir = l1 * d1 + l2 * d2
                heappush(self.shoot_queue,
                         (0.5 * (p1 + p2),
                          (new_dir, k_max + k, d1, k - 1, 0, d2, k, 0)))
                dirs_queue.append(new_dir)
        t_end = time.time()
        for e in self.shoot_queue:
            print(f'{e[0]:.5f}, {RAD_TO_DEG * e[1][0]:.2f}')
        t_start2 = time.time()

        try:
            while len(self.shoot_queue) != 0:
                self.iter += 1
                print(f"It {self.iter:>3} - ", end='')
                _, obj = heappop(self.shoot_queue)
                d, k, d_l, k_l, depth_l, d_u, k_u, depth_u = obj
                my_depth = max(depth_l, depth_u) + 1
                print(f'Shoot {f"{d * 180 / pi:.1f}":>4} - Depth {my_depth:>3} - ', end='')
                dirs_list.append(d)
                bl = False
                bu = False
                shoot = Shooting(self.mp.model.dyn, self.x_init, self.T, adapt_ts=self.adaptive_int_step,
                                 N_iter=self.N_iter, domain=self.mp.domain, fail_on_maxiter=True,
                                 factor=self.int_prec_factor, coords=self.mp.coords,
                                 target_crit=lambda x: self.distance(x, self.x_target) < self.opti_ceil)
                shoot.set_adjoint(self.adj_from_dir(d))
                traj = shoot.integrate()
                traj.label = self.iter
                self.trajs[k] = traj
                int_steps.append(traj.last_index)
                lp = np.zeros(2)
                lp[:] = traj.points[traj.last_index - 1]
                self.last_points[k] = lp
                d_target = self.distance(lp, self.x_target)
                d_ref_dist = self.distance(self.x_init, self.x_target)
                print(f'Crit {d_target:.1f}, {int(100 * d_target/d_ref_dist)}% - ', end='')
                b, index = traj.optimal, (traj.last_index - 1)  # self.is_opti(traj)
                if b:
                    print('Optimum found')
                    self.opti_indexes.append(k)
                    self.trajs[k].type = 'optimal'
                    self.opti_headings[k] = d
                    self.opti_time[k] = self.trajs[k].timestamps[index]
                    break
                lp_l = self.last_points[k_l]
                lp_u = self.last_points[k_u]
                if my_depth < max_depth:
                    if self.distance(lp, lp_l) > self.neighb_ceil:
                        bl = True
                        dist1 = min(self.distance(p, self.x_target) for p in traj.points[:traj.last_index])
                        dist2 = min(self.distance(p, self.x_target) for p in
                                 self.trajs[k_l].points[:self.trajs[k_l].last_index])
                        # d1 = self.distance(lp, self.x_target)
                        # d2 = self.distance(lp_l, self.x_target)
                        w1 = 0.5
                        w2 = 0.5
                        # w1 = dist2 / (dist1 + dist2)
                        # w2 = dist1 / (dist1 + dist2)

                        new_dir = w1 * d_l + w2 * d
                        new_prio = w1 * dist1 + w2 * dist2

                        heappush(self.shoot_queue,
                                 (new_prio, (new_dir, k_max, d_l, k_l, depth_l, d, k, my_depth)))
                        dirs_queue.append(new_dir)
                        k_max += 1
                    if self.distance(lp, lp_u) > self.neighb_ceil:
                        bu = True
                        dist1 = min(self.distance(p, self.x_target) for p in traj.points[:traj.last_index])
                        dist2 = min(self.distance(p, self.x_target) for p in
                                 self.trajs[k_u].points[:self.trajs[k_u].last_index])
                        # d1 = self.distance(lp, self.x_target)
                        # d2 = self.distance(lp_u, self.x_target)
                        w1 = 0.5
                        w2 = 0.5
                        # w1 = dist2 / (dist1 + dist2)
                        # w2 = dist1 / (dist1 + dist2)

                        new_dir = w1 * d + w2 * d_u
                        new_prio = w1 * dist1 + w2 * dist2

                        heappush(self.shoot_queue,
                                 (new_prio, (new_dir, k_max, d, k, my_depth, d_u, k_u, depth_u)))
                        dirs_queue.append(new_dir)
                        k_max += 1
                    if bl or bu:
                        print(f'Branching {("L" if bl else "") + ("U" if bu else ""):>2}')
                    else:
                        print('No branching')
                else:
                    print(f'Max depth ({max_depth}) reached, no branching')
        except KeyboardInterrupt:
            pass

        t_end2 = time.time()
        d_pre_proc = t_end - t_start
        d_proc = t_end2 - t_start2

        self.mp.trajs = list(self.trajs.values())
        print(f'    * Pre-processing :  {d_pre_proc:.3f} s')
        print(f'    * Processing      :  {d_proc:.3f} s')
        print(f'    * Total           :  {d_proc + d_pre_proc:.3f} s')
        if debug:
            plt.hist(180 / pi * np.array(dirs_list), np.linspace(0., 360., 180))
            plt.hist(180 / pi * np.array(dirs_queue), np.linspace(0., 360., 180))
            plt.show()
        if len(self.opti_indexes) == 0:
            print("No optimum found")
        else:
            for k in self.opti_indexes:
                print(f'         Optimal time : {self.opti_time[k]}')
                print(f'Optimal init. heading : {RAD_TO_DEG * self.opti_headings[k]}')
                self.trajs[k].type = 'optimal'

    def solve_fancy2(self, max_depth=10, debug=False):

        t_start = time.time()
        self.iter = 0
        int_steps = []
        k_max = len(self.dirs.keys())
        dirs_list = []
        dirs_queue = []
        k = 0

        def f(d):
            nonlocal k
            dirs_list.append(d)
            shoot = Shooting(self.mp.model.dyn, self.x_init, self.T, adapt_ts=self.adaptive_int_step,
                             N_iter=self.N_iter, domain=self.mp.domain, fail_on_maxiter=True,
                             factor=self.int_prec_factor, coords=self.mp.coords)
            shoot.set_adjoint(self.adj_from_dir(d).reshape((2,)))
            traj = shoot.integrate()
            self.trajs[k] = traj
            lp = np.zeros(2)
            lp[:] = traj.points[traj.last_index - 1]
            self.last_points[k] = lp
            int_steps.append(traj.last_index)
            val = np.mean(np.linalg.norm(traj.points[:traj.last_index] - self.x_target))
            k += 1
            return val

        t_end = time.time()
        t_start2 = time.time()
        try:
            res_list = []
            for k in range(len(self.dirs) - 1):
                res_list.append(scipy.optimize.minimize_scalar(f, bounds=(self.dirs[k], self.dirs[k + 1])))
            print(min(res_list, key=lambda x: x.fun))
        except KeyboardInterrupt:
            pass
        t_end2 = time.time()
        d_pre_proc = t_end - t_start
        d_proc = t_end2 - t_start2

        self.mp.trajs = list(self.trajs.values())
        print(f'    * Pre-processing :  {d_pre_proc:.3f} s')
        print(f'    * Processing      :  {d_proc:.3f} s')
        print(f'    * Total           :  {d_proc + d_pre_proc:.3f} s')
        if debug:
            plt.hist(180 / pi * np.array(dirs_list), np.linspace(0., 360., 180))
            plt.hist(180 / pi * np.array(dirs_queue), np.linspace(0., 360., 180))
            plt.show()
        if len(self.opti_indexes) == 0:
            print("No optimum found")
        else:
            for k in self.opti_indexes:
                print(f'         Optimal time : {self.opti_time[k]}')
                print(f'Optimal init. heading : {RAD_TO_DEG * self.opti_headings[k]}')
                self.trajs[k].type = 'optimal'

    def solve_global(self, gradient=False):

        # Preprocessing
        t_start = time.time()

        def f(d):
            shoot = Shooting(self.mp.model.dyn, self.x_init, self.T, adapt_ts=self.adaptive_int_step,
                             N_iter=self.N_iter, domain=self.mp.domain, fail_on_maxiter=True,
                             factor=self.int_prec_factor, coords=self.mp.coords)
            shoot.set_adjoint(self.adj_from_dir(d).reshape((2,)))
            traj = shoot.integrate()
            return self.distance(self.x_target, traj.points[-1])

        t_end = time.time()

        t_start2 = time.time()
        # res = scipy.optimize.shgo(f, ((self.min_init_angle, self.max_init_angle),))
        if gradient:
            for d in self.dirs.values():
                res = scipy.optimize.minimize(f, d, method='BFGS')
                shoot = Shooting(self.mp.model.dyn, self.x_init, self.T, adapt_ts=self.adaptive_int_step,
                                 N_iter=self.N_iter, domain=self.mp.domain, fail_on_maxiter=True,
                                 factor=self.int_prec_factor, coords=self.mp.coords)
                shoot.set_adjoint(self.adj_from_dir(res.x[0]).reshape((2,)))
                self.mp.trajs.append(shoot.integrate())
        else:
            res = scipy.optimize.shgo(f, ((self.min_init_angle, self.max_init_angle),))
        t_end2 = time.time()

        d_pre_proc = t_end - t_start
        d_proc = t_end2 - t_start2
        print(f'    * Pre-processing :  {d_pre_proc:.3f} s')
        print(f'    * Processing      :  {d_proc:.3f} s')
        print(f'    * Total           :  {d_proc + d_pre_proc:.3f} s')

        if not gradient:
            print(res)
            for i in range(len(res.x)):
                shoot = Shooting(self.mp.model.dyn, self.x_init, self.T, adapt_ts=self.adaptive_int_step,
                                 N_iter=self.N_iter, domain=self.mp.domain, fail_on_maxiter=True,
                                 factor=self.int_prec_factor, coords=self.mp.coords)
                shoot.set_adjoint(self.adj_from_dir(res.x[i]).reshape((2,)))
                self.mp.trajs.append(shoot.integrate())

    def shoot_init(self):
        for d in self.dirs.values():
            shoot = Shooting(self.mp.model.dyn, self.x_init, self.T, adapt_ts=self.adaptive_int_step,
                             N_iter=self.N_iter, domain=self.mp.domain, fail_on_maxiter=True,
                             factor=self.int_prec_factor, coords=self.mp.coords)
            shoot.set_adjoint(self.adj_from_dir(d).reshape((2,)))
            self.mp.trajs.append(shoot.integrate())
