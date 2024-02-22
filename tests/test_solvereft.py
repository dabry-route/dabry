import pytest

from dabry.problem import NavigationProblem
from dabry.solver_ef import SolverEFTrimming
from dabry.misc import Chrono


@pytest.mark.parametrize("pb_name", [k for k, v in NavigationProblem.ALL.items() if v['gh_wf'] == 'True'])
def test_solver(pb_name):
    pb = NavigationProblem.from_name(pb_name)
    solver = SolverEFTrimming(pb.rescale(), 1)
    with Chrono() as _:
        solver.solve()
