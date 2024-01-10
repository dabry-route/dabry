import argparse
import os.path
import sys

from dabry.misc import Chrono
from dabry.problem import all_problems
from dabry.solver_ef import SolverEFBase
from dabry.ddf_manager import DDFmanager


class Test:

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def solve(self, pb_id, debug=False, output=False):
        output_dir = os.path.join(self.output_dir, f'{pb_id}')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            for fname in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, fname))
        try:
            pb_class = list(all_problems.values())[pb_id][0]
            pb = pb_class()
            if output:
                pb.io.set_case(f'_test_{pb_id}')
                pb.io.clean_output_dir()
            t_upper_bound = pb.time_scale if pb.time_scale is not None else pb.l_ref / pb.model.srf
            solver_ef = SolverEFBase(pb, t_upper_bound, max_steps=700, rel_nb_ceil=0.02, quick_solve=args.quicksolve)
            res = solver_ef.solve(verbose=1 if args.multithreaded else 2)
            status = 0
            if output:
                pb.save_ff()
                extremals = solver_ef.get_trajs()
                pb.io.dump_trajs(extremals)
                pb.io.dump_trajs([res.traj])
                pb.save_info()
        except Exception as e:
            status = 1
            if debug:
                raise e
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
        if len(pb_nok) > 0:
            return 1
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test suite for solver')
    parser.add_argument('idtest', nargs='?', default=-1)
    parser.add_argument('-m', '--multithreaded', help='Run script with adapted verbosity', action='store_const',
                        const=True, default=False)
    parser.add_argument('-q', '--quicksolve', help='Use quick solve mode', action='store_const',
                        const=True, default=False)
    parser.add_argument('-s', '--sumup', help='Print test outcome sumup', action='store_const',
                        const=True, default=False)
    parser.add_argument('-d', '--debug', help='Debug mode', action='store_const',
                        const=True, default=False)
    parser.add_argument('-o', '--output', help='Output folder', action='store_const', const=True, default=False)
    args = parser.parse_args(sys.argv[1:])
    test_dir = os.path.dirname(__file__)
    unit_pb = -1
    pb_ok = []
    pb_nok = []
    problems = []
    try:
        os.mkdir(os.path.join(test_dir, 'out'))
    except FileExistsError:
        pass
    test = Test(os.path.join(test_dir, 'out'))
    if args.sumup:
        exit(test.sumup())
    if args.idtest != -1:
        unit_pb = int(sys.argv[1])
    else:
        problems = len(all_problems)

    if unit_pb >= 0:
        test.solve(unit_pb, args.debug, args.output)
    else:
        chrono = Chrono()
        chrono.start('Starting test suite')
        for pb_id in problems:
            test.solve(pb_id, args.debug)
        test.sumup()
