import os

from mermoz.mdf_manager import MDFmanager
from mermoz.obstacle import GreatCircleObs, ParallelObs, MaxiObs, LSEMaxiObs
from mermoz.params_summary import ParamsSummary
from mermoz.misc import *
from mermoz.problem import IndexedProblem, DatabaseProblem
from mermoz.solver_ef import SolverEF
from mermoz.solver_rp import SolverRP
import argparse

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Test suite for solver')
    # parser.add_argument('-t', '--test', help='Run given test')
    # args = parser.parse_args(sys.argv[1:])
    unit_pb = -1
    if len(sys.argv) > 1:
        unit_pb = int(sys.argv[1])
    pb_ok = []
    pb_nok = []


    def solve(pb_id):
        pb = IndexedProblem(pb_id)
        t_upper_bound = pb.time_scale if pb.time_scale is not None else pb.l_ref / pb.model.v_a
        solver_ef = SolverEF(pb, t_upper_bound, max_steps=700, rel_nb_ceil=0.02, quick_solve=False)
        solver_ef.solve()

    if unit_pb >= 0:
        solve(unit_pb)
    else:
        chrono = Chrono()
        chrono.start('Starting test suite')
        for pb_id in range(len(IndexedProblem.problems)):
            if pb_id in IndexedProblem.exclude_from_test:
                continue

            try:
                solve(pb_id)
                pb_ok.append(pb_id)
            except Exception:
                pb_nok.append(pb_id)
        chrono.stop()
        print(f'Passed ({len(pb_ok)}) : {pb_ok}')
        print(f'Failed ({len(pb_nok)}) : {pb_nok}')
