import argparse
import os.path
import sys
from datetime import datetime

from dabry.io_manager import IOManager
from dabry.misc import Utils, Chrono, Coords
from dabry.problem import NavigationProblem
from dabry.solver_ef import SolverEFResampling

"""
__main__.py
Module interface to CLI usage.

Copyright (C) 2021 Bastien Schnitzler 
(bastien dot schnitzler at live dot fr)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dabry trajectory planning', prog='dabry')
    subparsers = parser.add_subparsers(dest='dest')

    parser_idpb = subparsers.add_parser('case', help='Solve a given case from reference problems')
    parser_idpb.add_argument('name', help='Problem name')
    parser_idpb.add_argument('--airspeed', nargs='?', help='Vehicle speed relative to the flow in m/s', default=None)

    parser_real = subparsers.add_parser('real', help='Solve a given case from real data')
    parser_real.add_argument('x_init_lon', help='Initial point longitude in degrees')
    parser_real.add_argument('x_init_lat', help='Initial point latitude in degrees')
    parser_real.add_argument('x_target_lon', help='Initial point longitude in degrees')
    parser_real.add_argument('x_target_lat', help='Initial point latitude in degrees')
    parser_real.add_argument('start_date', help='Start date')
    parser_real.add_argument('airspeed', help='Airspeed in meters per seconds')
    parser_real.add_argument('pressure_level', help='Pressure level in hPa')
    parser_real.add_argument('--rft', help='Perform front tracking', action='store_const', const=True,
                             default=False)

    args = parser.parse_args(sys.argv[1:])
    if args.dest == 'case':
        pb_unscaled = NavigationProblem.from_name(args.name)

    else:
        x_init_deg, x_target_deg, start_date, airspeed, pressure_level = \
            Utils.process_pb_params(args.x_init_lon, args.x_init_lat, args.x_target_lon, args.x_target_lat,
                                    args.start_date, args.airspeed, args.pressure_level)
        x_init, x_target = Utils.DEG_TO_RAD * x_init_deg, Utils.DEG_TO_RAD * x_target_deg
        duration = 2 * Utils.distance(x_init, x_target, coords=Coords.GCS) / airspeed
        stop_date = datetime.fromtimestamp(start_date.timestamp() + duration)

        # Not an available parameter for the moment
        resolution = '0.5'

        cds_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cds'))
        IOManager.query_era5(start_date, stop_date, cds_dir, pressure_level=pressure_level, resolution=resolution)
        pb_unscaled = NavigationProblem.from_database(x_init, x_target, airspeed, start_date.timestamp(),
                                                      stop_date.timestamp(), resolution=resolution,
                                                      pressure_level=pressure_level, data_path=cds_dir)

    pb = pb_unscaled.rescale()

    solver = SolverEFResampling(pb)

    with Chrono() as _:
        solver.solve()

    print(solver.success)

    traj_ortho = pb.orthodromic()
    traj_radial = pb.radial()

    pb.io.set_case_dir(os.path.join(os.path.abspath('.'), pb_unscaled.name))
    pb.io.clean_output_dir()
    _, _, _, _, _, scale_length, scale_time = pb_unscaled.scaling_params()
    scaling_params = dict(scale_length=scale_length, scale_time=scale_time, bl=pb_unscaled.bl,
                          time_offset=pb_unscaled.model.ff.t_start)
    solver.save_results(**scaling_params)
    pb_unscaled.io.set_case_dir(os.path.join(os.path.abspath('.'), pb_unscaled.name))
    pb_unscaled.io.save_ff(pb_unscaled.model.ff, bl=pb_unscaled.bl, tr=pb_unscaled.tr)
    pb_unscaled.save_info()
    pb_unscaled.io.save_traj(traj_ortho, 'orthodromic', **scaling_params)
    pb_unscaled.io.save_traj(traj_radial, 'radial', **scaling_params)
    print(f'Results saved to {pb.io.case_dir}')
