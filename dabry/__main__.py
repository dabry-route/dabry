import argparse
import os.path
import sys
from datetime import datetime

from dabry.io_manager import IOManager
from dabry.misc import Utils, Chrono
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
        pb = pb_unscaled.rescale()

        solver = SolverEFResampling(pb)

        with Chrono() as _:
            solver.solve()

        print(solver.success)

        traj_ortho = pb.orthodromic()
        traj_htarget = pb.htarget()

        pb.io.clean_output_dir()
        solver.save_results()
        pb.save_info()
        print(f'Results saved to {pb.io.case_dir}')

    else:
        x_init_deg, x_target_deg, start_date, airspeed, pressure_level = \
            Utils.process_pb_params(args.x_init_lon, args.x_init_lat, args.x_target_lon, args.x_target_lat,
                                    args.start_date, args.airspeed, args.pressure_level)
        x_init, x_target = Utils.DEG_TO_RAD * x_init_deg, Utils.DEG_TO_RAD * x_target_deg
        duration = 2 * Utils.distance(x_init, x_target, coords=Utils.COORD_GCS) / airspeed
        stop_date = datetime.fromtimestamp(start_date.timestamp() + duration)

        # Not an available parameter for the moment
        resolution = '0.5'

        cds_dir = os.path.join('data', 'cds')
        IOManager.query_era5(start_date, stop_date, cds_dir, pressure_level=pressure_level, resolution=resolution)
        pb = NavigationProblem.from_database(x_init, x_target, airspeed, start_date.timestamp(), stop_date.timestamp(),
                                             resolution=resolution, pressure_level=pressure_level, data_path=cds_dir)

        case_name = IOManager.format_cname(x_init_deg, x_target_deg, start_date.timestamp())
        pb.io.set_case('main_' + case_name)
        pb.io.clean_output_dir()

        pb.save_ff()

        chrono = Chrono()

        # Setting the extremal solver
        solver_ef = solver = SolverEFBase(pb, pb.time_scale, max_steps=700, rel_nb_ceil=0.02, quick_solve=True)

        chrono.start('Solving problem using extremal field (EF)')
        res_ef = solver_ef.solve()
        chrono.stop()
        if res_ef.status:
            # Solution found
            # Save optimal trajectory
            pb.io.dump_trajs([res_ef.traj])
            print(f'Target reached in : {Utils.time_fmt(res_ef.duration)}')
        else:
            print('No solution found')

        extremals = solver_ef.get_trajs()
        pb.io.dump_trajs(extremals)
        # pb.io.dump_obs(pb, nx_rft, ny_rft)

        pb.orthodromic()
        # mdfm.dump_trajs([pb.trajs[-1]])

        pb.save_info()
        # Also copy the script that produced the result to output dir for later reproduction
        pb.io.save_script(__file__)

        print(f'Results saved to {pb.io.case_dir}')
