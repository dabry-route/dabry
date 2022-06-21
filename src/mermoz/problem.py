import numpy as np
from mpl_toolkits.basemap import Basemap
from numpy import ndarray

from mermoz.feedback import Feedback
from mermoz.integration import IntEulerExpl
from mermoz.mdf_manager import MDFmanager
from mermoz.model import Model
from mermoz.stoppingcond import StoppingCond, TimedSC
from mermoz.misc import *


class MermozProblem:
    """
    The Mermoz problem class defines the whole optimal control problem.

    The UAV is assimilated to a point in 2D space. Its starting point is assumed at (0, 0).
    Its target is located at (x_f, 0) (coordinate system is adapted to the target).
    """

    def __init__(self,
                 model: Model,
                 coords=None,
                 domain=None,
                 T=0.,
                 visual_mode="full",
                 axes_equal=True,
                 autodomain=True,
                 mask_land=True):
        self.model = model

        self.coords = coords
        self.bm = None
        if not domain:
            if not autodomain:
                self.domain = lambda _: True
            else:
                # Bound computation domain on wind grid limits
                wind = self.model.wind
                bl = (wind.x_min, wind.y_min)
                tr = (wind.x_max, wind.y_max)
                if self.coords == COORD_GCS and mask_land:
                    factor = 1. if wind.units_grid == U_RAD else RAD_TO_DEG
                    self.bm = Basemap(llcrnrlon=factor * bl[0],
                                      llcrnrlat=factor * bl[1],
                                      urcrnrlon=factor * tr[0],
                                      urcrnrlat=factor * tr[1],
                                      projection='cyl', resolution='c')
                self.domain = lambda x: bl[0] < x[0] < tr[0] and bl[1] < x[1] < tr[1] and \
                                        (self.bm is None or not self.bm.is_land(factor * x[0], factor * x[1]))
        else:
            self.domain = domain
        self._feedback = None
        title = "$v_a=" + str(self.model.v_a) + "\:m/s$, $T=" + str(T) + "\:s$"
        self.trajs = []

    def __str__(self):
        return str(self.model)

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
        sc = TimedSC(1.)
        sc.value = lambda t, x: stop_cond.value(t, x) or not self.domain(x)
        integrator = IntEulerExpl(self.model.wind,
                                  self.model.dyn,
                                  self._feedback,
                                  self.coords,
                                  stop_cond=sc,
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
                e_minuses = np.array([self.model.wind.e_minus(x) for x in traj.points])
                norms = 1 / np.linalg.norm(e_minuses, axis=1)
                e_minuses = np.einsum('ij,i->ij', e_minuses, norms)
                scalar_prod = np.abs(np.einsum('ij,ij->i', vect_controls, e_minuses))
            self.display.plot_traj(traj, color_mode=color_mode, controls=False, scalar_prods=scalar_prod)
