import argparse
import sys

from dabry.problem import NavigationProblem
from dabry.solver_ef import SolverEFResampling


def main(pb_name):
    pb = NavigationProblem.from_name(pb_name)

    solver = SolverEFResampling(pb.rescale(), 1, n_time=100, rel_max_step=1)

    solver.setup()
    solver.solve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SolverEFResampling on problem database')
    parser.add_argument('name')
    args = parser.parse_args(sys.argv[1:])
    main(args.name)
