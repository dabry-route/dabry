import argparse
import os.path
import shutil
import sys

from mermoz.misc import *
from mermoz.problem import IndexedProblem
from mermoz.solver_ef import SolverEF


class Test:

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def solve(self, pb_id):
        output_dir = os.path.join(self.output_dir, f'{pb_id}')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            for fname in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, fname))
        try:
            pb = IndexedProblem(pb_id)
            t_upper_bound = pb.time_scale if pb.time_scale is not None else pb.l_ref / pb.model.v_a
            solver_ef = SolverEF(pb, t_upper_bound, max_steps=700, rel_nb_ceil=0.02, quick_solve=args.quicksolve)
            solver_ef.solve(verbose=1 if args.multithreaded else 2)
            status = 0
        except Exception:
            status = 1
        with open(os.path.join(output_dir, f'{pb_id}'), 'w') as f:
            f.writelines([f'{status}'])

    def sumup(self):
        outs = os.listdir(self.output_dir)
        pb_ok = []
        pb_nok = []
        for fname in outs:
            with open(os.path.join(self.output_dir, fname, fname), 'r') as f:
                s = f.readline()
                if int(s[0]) == 0:
                    pb_ok.append(int(fname))
                else:
                    pb_nok.append(int(fname))
        print(f'Passed ({len(pb_ok)}) : {sorted(pb_ok)}')
        print(f'Failed ({len(pb_nok)}) : {sorted(pb_nok)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test suite for solver')
    parser.add_argument('idtest', nargs='?', default=-1)
    parser.add_argument('-m', '--multithreaded', help='Run script with adapted verbosity', action='store_const',
                        const=True, default=False)
    parser.add_argument('-q', '--quicksolve', help='Use quick solve mode', action='store_const',
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

    test = Test('tests/out')

    if unit_pb >= 0:
        test.solve(unit_pb)
    else:
        chrono = Chrono()
        chrono.start('Starting test suite')
        for pb_id in problems:
            test.solve(pb_id)
        test.sumup()
