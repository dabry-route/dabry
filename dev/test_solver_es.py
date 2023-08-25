
from dabry.solver_es import SolverES
from dabry.ddf_manager import DDFmanager
from dabry.problem import IndexedProblem

pb = IndexedProblem('sanjuan-dublin-ortho-tv')

solver = SolverES(pb)
traj = solver.solve(20*3600)

ddf = DDFmanager()
ddf.setup()
ddf.set_case('test_solver_es')
ddf.clean_output_dir()
ddf.dump_wind(pb.model.wind)
ddf.dump_trajs([traj])
ddf.log(pb)