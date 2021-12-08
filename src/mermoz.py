import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from src.feedback import Feedback, RandomFB
from src.integration import IntEulerExpl
from src.model import Model
from src.stoppingcond import StoppingCond, TimedSC
from src.visual import Visual


class MermozProblem:
    """
    The Mermoz problem class defines the whole optimal control problem.

    The UAV is assimilated to a point in 2D space. Its starting point is assumed at (0, 0).
    Its target is located at (x_f, 0) (coordinate system is adapted to the target).
    """

    def __init__(self,
                 model: Model,
                 max_iter=10000,
                 int_step=0.0001,
                 T=0.,
                 visual_mode="full"):
        self._model = model
        self._feedback = None
        title = "$v_a=" + str(self._model.v_a) + "\:m/s$, $T=" + str(T) + "\:s$"
        self.display = Visual(visual_mode,
                              2,
                              1,
                              lambda x: self._model.wind.value(x) / self._model.v_a,
                              title=title)
        self.trajs = []

        self.plot_modes = ["default", "reachability"]

    def load_feedback(self, feedback: Feedback):
        """
        Load a feedback law for integration

        :param feedback: A feedback law
        """
        self._feedback = feedback

    def reachability(self,
                     T: float,
                     N_samples=100,
                     max_iter=20000,
                     int_step=0.01):
        """
        Compute sample trajectories for a reachability analysis of the problem

        :param T: Stop time
        :param N_samples: Number of sample trajectories
        """
        for ns in range(N_samples):
            self.load_feedback(RandomFB(-np.pi, np.pi, seed=42 + ns))
            stop_cond = TimedSC(T)
            self.integrate_trajectory(stop_cond, max_iter=max_iter, int_step=int_step)

    def stop_cond(self, x: ndarray):
        """
        Provide a stopping condition for integration
        :return: A boolean telling whether to stop the integration or not given the state
        """
        return x[0] < self._model.x_f

    def integrate_trajectory(self,
                             stop_cond: StoppingCond,
                             max_iter=20000,
                             int_step=0.0001):
        if self._feedback is None:
            raise ValueError("No feedback provided for integration")
        integrator = IntEulerExpl(self._model.wind,
                                  self._model.dyn,
                                  self._feedback,
                                  stop_cond=stop_cond,
                                  max_iter=max_iter,
                                  int_step=int_step)
        self.trajs.append(integrator.integrate(np.array([0., 0.])))

    def plot_trajs(self, mode="default"):
        if mode not in self.plot_modes:
            raise ValueError(f"Unknown plot mode : {mode}")
        for traj in self.trajs:
            # print(traj.adjoints[0])
            self.display.plot_traj(traj, mode=mode)
        plt.show()
