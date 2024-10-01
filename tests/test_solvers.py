import pytest

from dabry.problem import NavigationProblem
from dabry.solver_ef import SolverEFTrimming, SolverEF


@pytest.mark.parametrize("pb_name", [k for k, v in NavigationProblem.ALL.items() if v['gh_wf'] == 'True'])
@pytest.mark.parametrize("solverclass", [SolverEF, SolverEFTrimming])
def test_solver(solverclass, pb_name):
    pb_unscaled = NavigationProblem.from_name(pb_name)
    pb = pb_unscaled.rescale()
    tb_txt = NavigationProblem.ALL[pb_name]['time_bound']
    time_bound = float(tb_txt) if len(tb_txt) > 0 else None
    args = (pb, time_bound) if time_bound is not None else (pb,)
    solver = solverclass(*args)
    solver.solve()
    if not solver.success:
        raise Exception('Problem not solved')
