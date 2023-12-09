import numpy as np

from dabry.aero import Aero
from dabry.misc import Chrono
from dabry.problem import NavigationProblem
from dabry.solver_ef import SolverEF, Pareto


class EnergyAnalysis:

    def __init__(self, pb: NavigationProblem):
        self.pb = pb
        self.pareto = Pareto()

    def _airspeed_fromindex(self, i):
        k = np.floor(np.log2(i + 1))
        m = i + 1 - 2 ** k
        v = 0.5 + (-1) ** (m + 1) * ((1 + 2 * (m // 2)) / (2 ** (k + 1)))
        return self.pb.aero.v_min + v * (self.pb.aero.v_max - self.pb.aero.v_min)

    def _airspeed_range(self, imin, imax):
        return list(map(self._airspeed_fromindex, np.arange(imin, imax)))

    def airspeed_range(self, *args):
        if len(args) == 1:
            return self._airspeed_range(0, args[0])
        elif len(args) == 2:
            return self._airspeed_range(*args)
        return None

    def opt_traj(self, asp, mode, ddf=None, dt=None):
        chrono = Chrono()
        self.pb.update_airspeed(asp)
        chrono.start(f'{"TIMEOPT" if not mode else "ENEROPT"} with airspeed {asp:.2f}')
        if mode == 0:
            kwargs = {
                'max_steps': 500,
                'quick_solve': True,
                'no_coll_filtering': True,
            }
        else:
            kwargs = {
                'max_steps': 500,
                'mode': 1,
                'pareto': self.pareto,
                'no_coll_filtering': True,
                'quick_solve': True,
            }
        if dt is None:
            kwargs['max_time'] = self.pb.time_scale
        else:
            kwargs['dt'] = dt
        solver = SolverEF(self.pb, **kwargs)
        res_ef = solver.solve(verbose=2)
        chrono.stop()
        if ddf is not None:
            ddf.dump_trajs(solver.get_trajs())
        return res_ef

    def sample_traj(self, mode, N=10, dt=None):
        """
        Sample the energy diagram with constant airspeed optimal trajectories
        :param N: Number of different airspeed to compute between min and max
        :return:
        """
        list_fail = []
        list_res = []
        asps = self.airspeed_range(N)
        for k, v_a in enumerate(asps):
            res_ef = self.opt_traj(v_a, mode, dt=dt)
            if not res_ef.status:
                print(f'No solution found for {k}')
            else:
                list_res.append(res_ef)
        return list_res
