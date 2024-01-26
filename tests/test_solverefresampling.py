import argparse
import sys

from dabry.problem import NavigationProblem
from dabry.solver_ef import SolverEFResampling


def main(pb_name):
    pb = NavigationProblem.from_name(pb_name)

    # ff_disc = DiscreteFF.from_ff(pb.model.ff, (pb.bl, pb.tr))
    # pb.update_ff(ff_disc)

    solver = SolverEFResampling(pb.rescale(), 0.8, n_time=100)

    solver.setup()
    solver.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SolverEFResampling on problem database')
    parser.add_argument('name')
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
    main(args.name)