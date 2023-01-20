import argparse

from mermoz.misc import *
from mermoz.problem import IndexedProblem
from mermoz.solver_ef import SolverEF

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test suite for solver')
    parser.add_argument('idtest', nargs='?', default=-1)
    parser.add_argument('-m', '--multithreaded', help='Run script with adapted verbosity', action='store_const',
                        const=True, default=False)
    args = parser.parse_args(sys.argv[1:])
    unit_pb = -1
    pb_ok = []
    pb_nok = []
    problems = []
    if args.idtest != -1:
        unit_pb = int(sys.argv[1])
    else:
        problems = list(IndexedProblem.problems.keys())
        for pb_id in IndexedProblem.exclude_from_test:
            problems.remove(pb_id)

    def solve(pb_id):
        pb = IndexedProblem(pb_id)
        t_upper_bound = pb.time_scale if pb.time_scale is not None else pb.l_ref / pb.model.v_a
        solver_ef = SolverEF(pb, t_upper_bound, max_steps=700, rel_nb_ceil=0.02, quick_solve=False)
        solver_ef.solve(verbose=1 if args.multithreaded else 2)

    if unit_pb >= 0:
        solve(unit_pb)
    else:
        chrono = Chrono()
        chrono.start('Starting test suite')
        for pb_id in problems:
            try:
                solve(pb_id)
                pb_ok.append(pb_id)
            except Exception:
                pb_nok.append(pb_id)
        chrono.stop()
        print(f'Passed ({len(pb_ok)}) : {pb_ok}')
        print(f'Failed ({len(pb_nok)}) : {pb_nok}')
