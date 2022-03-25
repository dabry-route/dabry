import numpy as np
from numpy import ndarray

from mermoz.feedback import Feedback
from mermoz.integration import IntEulerExpl
from mermoz.model import Model
from mermoz.stoppingcond import StoppingCond
from mermoz.visual import Visual


class MermozProblem:
    """
    The Mermoz problem class defines the whole optimal control problem.

    The UAV is assimilated to a point in 2D space. Its starting point is assumed at (0, 0).
    Its target is located at (x_f, 0) (coordinate system is adapted to the target).
    """

    def __init__(self,
                 model: Model,
                 coords='cartesian',
                 domain=None,
                 T=0.,
                 visual_mode="full",
                 axes_equal=True):
        self._model = model

        self.coords = coords
        if not domain:
            self.domain = lambda _: True
        else:
            self.domain = domain
        self._feedback = None
        title = "$v_a=" + str(self._model.v_a) + "\:m/s$, $T=" + str(T) + "\:s$"
        self.display = Visual(visual_mode,
                              2,
                              1,
                              lambda x: self._model.wind.value(x) / self._model.v_a,
                              title=title,
                              axes_equal=axes_equal)
        self.trajs = []

    def __str__(self):
        return str(self._model)

    def load_feedback(self, feedback: Feedback):
        """
        Load a feedback law for integration

        :param feedback: A feedback law
        """
        self._feedback = feedback

    def integrate_trajectory(self,
                             x_init: ndarray,
                             stop_cond: StoppingCond,
                             max_iter=20000,
                             int_step=0.0001):
        """
        Use the specified discrete integration method to build a trajectory
        with the given control law. Store the integrated trajectory in
        the trajectory list.
        :param x_init: The initial state
        :param stop_cond: A stopping condition for the integration
        :param max_iter: Maximum number of iterations
        :param int_step: Integration step
        """
        if self._feedback is None:
            raise ValueError("No feedback provided for integration")
        integrator = IntEulerExpl(self._model.wind,
                                  self._model.dyn,
                                  self._feedback,
                                  stop_cond=stop_cond,
                                  max_iter=max_iter,
                                  int_step=int_step)
        self.trajs.append(integrator.integrate(x_init))

    def eliminate_trajs(self, target, tol: float):
        """
        Delete trajectories that are too far from the objective point
        :param tol: The radius tolerance around the objective in meters
        """
        delete_index = []
        for k, traj in enumerate(self.trajs):
            keep = False
            for id, p in enumerate(traj.points):
                if np.linalg.norm(p - target) < tol:
                    keep = True
                    break
            if not keep:
                delete_index.append(k)
        for index in sorted(delete_index, reverse=True):
            del self.trajs[index]

    def plot_trajs(self, color_mode="default"):
        for traj in self.trajs:
            # print(traj.adjoints[0])
            scalar_prod = None
            if color_mode == 'reachability-enhanced':
                vect_controls = np.array([np.array([np.cos(u), np.sin(u)]) for u in traj.controls])
                e_minuses = np.array([self._model.wind.e_minus(x) for x in traj.points])
                norms = 1 / np.linalg.norm(e_minuses, axis=1)
                e_minuses = np.einsum('ij,i->ij', e_minuses, norms)
                scalar_prod = np.abs(np.einsum('ij,ij->i', vect_controls, e_minuses))
            self.display.plot_traj(traj, color_mode=color_mode, controls=False, scalar_prods=scalar_prod)
