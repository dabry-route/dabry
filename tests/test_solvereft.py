import argparse
import sys

from dabry.problem import NavigationProblem
from dabry.solver_display import display
from dabry.solver_ef import SolverEFTrimming
from dabry.misc import Chrono


def main(pb_name, disp=False):
    pb = NavigationProblem.from_name(pb_name)

    solver = SolverEFTrimming(pb.rescale(), 1, n_time=101)

    with Chrono() as _:
        solver.solve()

    if disp:
        display(solver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SolverEFTrimming on problem database')
    parser.add_argument('name')
    parser.add_argument('-d', '--display', action='store_const', default=False, const=True)
    args = parser.parse_args(sys.argv[1:])
    main(args.name, args.display)
