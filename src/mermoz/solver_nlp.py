import numpy as np
from mermoz.problem import MermozProblem
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from math import atan2


class SolverNLP:
    def __init__(self,
                 mp: MermozProblem,
                 T,
                 nt,
                 i_guess=None):
        self.mp = mp
        self.x_init = np.zeros(2)
        self.x_init[:] = mp.x_init
        self.x_target = np.zeros(2)
        self.x_target[:] = mp.x_target
        self.T = T
        self.nt = nt
        if i_guess is not None:
            self.i_guess = np.zeros(i_guess.shape)
            self.i_guess[:] = i_guess
        else:
            self.i_guess = np.zeros(3 * self.nt)
            for k in range(nt):
                self.i_guess[k] = self.x_init[0]
            for k in range(nt):
                self.i_guess[nt + k] = self.x_init[1]
            for k in range(nt - 1):
                self.i_guess[2 * nt + k] = 3.5
        self.dt = self.T / (self.nt - 1)
        self.ts = np.linspace(0, T, self.nt)

    def shoot(self, u):
        x = np.zeros((self.nt, 2))
        x[:] = self.x_init
        for k in range(self.nt - 1):
            x[k + 1, :] = x[k] + self.dt * self.mp.model.dyn.value(x[k], u[k], self.ts[k])
        return x

    def loss(self, u):
        x = np.zeros((self.nt, 2))
        x[:] = self.shoot(u)
        nt = self.nt
        return ((x[nt - 1, 0] - self.x_target[0]) ** 2 + (x[nt - 1, 1] - self.x_target[1]) ** 2) / 1e12

    def solve(self):
        nt = self.nt
        # res = minimize(lambda x: (x[nt - 1] - self.x_target[0]) ** 2 + (x[2 * nt - 1] - self.x_target[1]) ** 2,
        #                self.i_guess,
        #                method='SLSQP',
        #                constraints=constraints,
        #                options={'ftol': 1e-4, 'disp': True},
        #                bounds=Bounds(lb, ub))
        # constraints = []
        # for k in range(self.nt - 1):
        #     constraints += [
        #         {'type': 'ineq',
        #          'fun': lambda u: u[k] ** 2 + u[k + self.nt - 1] ** 2,
        #          'ub': 1.,
        #          }
        #     ]
        res = minimize(self.loss,
                       self.i_guess,
                       method='SLSQP',
                       #constraints=constraints,
                       options={'ftol': 1e-8, 'disp': True, 'maxiter': 200},
                       bounds=[(-6.28, 6.28) for _ in range(self.nt - 1)])
        return res, self.shoot(res.x)

    def solve_not_working(self):
        nt = self.nt
        constraints = []
        for k in range(self.nt - 1):
            # constraints += [
            #     {'type': 'eq',
            #      'fun':
            #          lambda x: x[k + 1] - x[k] + self.dt * self.mp.model.dyn.value(np.array((x[k], x[k + nt])),
            #                                                                        x[k + 2 * nt], self.ts[k])[0]
            #      },
            #     {'type': 'eq',
            #      'fun':
            #          lambda x: x[nt + k + 1] - x[nt + k] + self.dt *
            #                    self.mp.model.dyn.value(np.array((x[k], x[k + nt])),
            #                                            x[k + 2 * nt], self.ts[k])[1]
            #      }
            # ]
            constraints += [
                {'type': 'ineq',
                 'fun':
                     lambda x: x[k + 1] - x[k] + self.dt * self.mp.model.dyn.value(np.array((x[k], x[k + nt])),
                                                                                   x[k + 2 * nt], self.ts[k])[0],
                 'ub': 1e1,
                 'lb': -1e1
                 },
                {'type': 'ineq',
                 'fun':
                     lambda x: x[nt + k + 1] - x[nt + k] + self.dt *
                               self.mp.model.dyn.value(np.array((x[k], x[k + nt])),
                                                       x[k + 2 * nt], self.ts[k])[1],
                 'ub': 1e1,
                 'lb': -1e1
                 }
            ]
        lb = []
        ub = []
        wbbox = abs(self.x_target[0] - self.x_init[0])
        hbbox = abs(self.x_target[1] - self.x_init[1])
        sl = 1.2 * max(hbbox, wbbox)
        center = 0.5 * (self.x_init + self.x_target)
        for _ in range(nt):
            lb.append(center[0] - 0.5 * sl)
            ub.append(center[0] + 0.5 * sl)
        for _ in range(nt):
            lb.append(center[1] - 0.5 * sl)
            ub.append(center[1] + 0.5 * sl)
        for _ in range(nt - 1):
            lb.append(-100.)
            ub.append(100.)
        # res = minimize(lambda x: (x[nt - 1] - self.x_target[0]) ** 2 + (x[2 * nt - 1] - self.x_target[1]) ** 2,
        #                self.i_guess,
        #                method='SLSQP',
        #                constraints=constraints,
        #                options={'ftol': 1e-4, 'disp': True},
        #                bounds=Bounds(lb, ub))
        res = minimize(lambda x: (x[nt - 1] - self.x_target[0]) ** 2 + (x[2 * nt - 1] - self.x_target[1]) ** 2,
                       self.i_guess,
                       method='SLSQP',
                       constraints=constraints,
                       options={'ftol': 1e2, 'disp': True, 'maxiter': 200, 'eps': self.mp._geod_l / 1000},
                       bounds=Bounds(lb, ub))
        return res
