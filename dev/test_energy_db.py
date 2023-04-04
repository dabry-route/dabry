from datetime import datetime
from dabry.aero import MermozAero
from dabry.energy import EnergyAnalysis
from dabry.ddf_manager import DDFmanager
from dabry.misc import Utils, Chrono
from dabry.problem import DatabaseProblem

if __name__ == '__main__':

    pb_params = '-35.16 -7.71 -14.24 11.52 202111011200 23.0 500'
    x_init, x_target, start_date, airspeed, level = Utils.read_pb_params(pb_params)

    duration = 2 * Utils.distance(Utils.DEG_TO_RAD * x_init, Utils.DEG_TO_RAD * x_target,
                                  coords=Utils.COORD_GCS) / airspeed
    stop_date = datetime.fromtimestamp(start_date.timestamp() + duration)

    ddf = DDFmanager()
    ddf.setup()

    ddf.retrieve_wind(start_date, stop_date, level=level, res='0.5')
    case_name = '_test_energy'

    cache_wind = False
    cache_rff = False

    # This instance prints absolute elapsed time between operations
    chrono = Chrono()

    # Create a file manager to dump problem data
    mdfm = DDFmanager(cache_wind=cache_wind, cache_rff=cache_rff)
    mdfm.setup()
    mdfm.set_case(case_name)
    #mdfm.clean_output_dir()

    # Space and time discretization
    # Will be used to save wind when wind is analytical and shall be sampled
    # Will also be used by front tracking module
    nx_rft = 101
    ny_rft = 101
    nt_rft = 20

    pb = DatabaseProblem(x_init=Utils.DEG_TO_RAD * x_init,
                         x_target=Utils.DEG_TO_RAD * x_target, airspeed=airspeed, t_start=start_date.timestamp(),
                         t_end=stop_date.timestamp(), altitude=level, resolution='0.5')

    # pb.flatten()

    if not cache_wind:
        chrono.start('Dumping windfield to file')
        mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
        chrono.stop()

    aero = MermozAero()
    ea = EnergyAnalysis(pb, aero)

    # list_res = ea.sample_constant_asp(8)
    # for res in list_res:
    #     mdfm.dump_trajs([res.traj], filename='trajectories_timeopt.h5')

    ea.pareto.load_from_trajs(mdfm.case_dir, aero=aero, filename='trajectories_timeopt.h5')

    list_res = ea.sample_eneropt(4)
    for res in list_res:
        mdfm.dump_trajs([res.traj], filename='trajectories_eneropt.h5')

    # Extract information for display and write it to output
    mdfm.ps.load_from_problem(pb)
    mdfm.ps.dump()
    # Also copy the script that produced the result to output dir for later reproduction
    mdfm.save_script(__file__)

    # mdfm.set_case('example_dakar-natal-constr*')
    # mdfm.dump_trajs([res_ef.traj])

    print(f'Results saved to {mdfm.case_dir}')