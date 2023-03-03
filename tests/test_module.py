import argparse
import os.path
import sys

from dabry.misc import Chrono
from dabry.problem import IndexedProblem
from dabry.solver_ef import SolverEF
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
            ddf = DDFmanager()
            if output:
                ddf.setup()
                ddf.set_case(f'_test_{pb_id}')
                ddf.clean_output_dir()
            pb = IndexedProblem(pb_id)
            t_upper_bound = pb.time_scale if pb.time_scale is not None else pb.l_ref / pb.model.v_a
            solver_ef = SolverEF(pb, t_upper_bound, max_steps=700, rel_nb_ceil=0.02, quick_solve=args.quicksolve)
            res = solver_ef.solve(verbose=1 if args.multithreaded else 2)
            status = 0
            if output:
                ddf.dump_wind(pb.model.wind, nx=101, ny=101, nt=10, bl=pb.bl, tr=pb.tr, coords=pb.coords)
                extremals = solver_ef.get_trajs()
                ddf.dump_trajs(extremals)
                ddf.dump_trajs([res.traj])
                ddf.ps.load_from_problem(pb)
                ddf.ps.dump()
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
        test.sumup()
        exit(0)
    if args.idtest != -1:
        unit_pb = int(sys.argv[1])
    else:
        problems = list(range(len(IndexedProblem.problems)))

    if unit_pb >= 0:
        test.solve(unit_pb, args.debug, args.output)
    else:
        chrono = Chrono()
        chrono.start('Starting test suite')
        for pb_id in problems:
            test.solve(pb_id, args.debug)
        test.sumup()
