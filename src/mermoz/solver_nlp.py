import numpy as np
from mermoz.problem import MermozProblem
from scipy.optimize import Bounds, NonlinearConstraint, minimize


class SolverNLP:
    def __init__(self,
                 mp: MermozProblem,
                 x_init,
                 x_target,
                 T,
                 nt,
                 i_guess):
        self.mp = mp
        self.x_init = np.zeros(2)
        self.x_init[:] = x_init
        self.x_target = np.zeros(2)
        self.x_target[:] = x_target
        self.T = T
        self.nt = nt
        self.i_guess = np.zeros(3*self.nt)
        for k in range(nt):
            self.i_guess[k] = self.x_init[0]
        for k in range(nt):
            self.i_guess[nt + k] = self.x_init[1]
        for k in range(nt):
            self.i_guess[2*nt + k] = 3.5
        self.dt = self.T / (self.nt - 1)
        self.ts = np.linspace(0, T, self.nt)

    def solve(self):
        nt = self.nt
        constraints = []
        for k in range(self.nt - 1):
            constraints += [
                {'type': 'eq',
                 'fun':
                     lambda x: x[k + 1] - x[k] + self.dt * self.mp.model.dyn.value(np.array((x[k], x[k + nt])),
                                                                                   x[k + 2 * nt], self.ts[k])[0]
                 },
                {'type': 'eq',
                 'fun':
                     lambda x: x[nt + k + 1] - x[nt + k] + self.dt * self.mp.model.dyn.value(np.array((x[k], x[k + nt])),
                                                                                   x[k + 2 * nt], self.ts[k])[1]
                 }
            ]
        lb = []
        ub = []
        for _ in range(nt):
            lb.append(self.x_target[0]-0.01)
            ub.append(self.x_init[0]+0.01)
        for _ in range(nt):
            lb.append(self.x_target[1]-0.01)
            ub.append(self.x_init[1]+0.01)
        for _ in range(nt):
            lb.append(-4.)
            ub.append(4.)
        res = minimize(lambda x: (x[nt-1] - self.x_target[0]) ** 2 + (x[2*nt-1] - self.x_target[1]) ** 2,
                       self.i_guess,
                       method='SLSQP',
                       constraints=constraints,
                       options={'ftol': 1e-9, 'disp': True},
                       bounds=Bounds(lb, ub))
        print(res)
