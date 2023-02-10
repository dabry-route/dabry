import os
from dabry.ddf_manager import DDFmanager
from dabry.misc import Chrono
from dabry.problem import IndexedProblem, DatabaseProblem

if __name__ == '__main__':
    dbpb = 'ncdc/30E_28N_133E_1N_20220301_12'  # '37W_8S_16W_17S_20220301_12'

    chrono = Chrono()

    # Create a file manager to dump problem data
    ddfm = DDFmanager()
    ddfm.setup()
    ddfm.set_case(f'wind_{dbpb}')
    ddfm.clean_output_dir()

    nx_rft = 101
    ny_rft = 101
    nt_rft = 50

    pb = DatabaseProblem(dbpb)

    chrono.start('Dumping windfield to file')
    ddfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft)
    chrono.stop()
