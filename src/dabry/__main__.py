import argparse
import os.path
import sys
from datetime import datetime

import numpy as np

from dabry.ddf_manager import DDFmanager
from dabry.misc import Utils, Chrono
from dabry.problem import IndexedProblem, DatabaseProblem
from dabry.solver_ef import SolverEF
from dabry.solver_rp import SolverRP

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
    subparsers = parser.add_subparsers()

    parser_idpb = subparsers.add_parser('case', help='Solve a given case from reference problems')
    parser_idpb.add_argument('test_id', default=-1, help='Problem id (int) or name (string)')

    parser_real = subparsers.add_parser('real', help='Solve a given case from real data')
    parser_real.add_argument('x_init_lon', help='Initial point longitude in degrees')
    parser_real.add_argument('x_init_lat', help='Initial point latitude in degrees')
    parser_real.add_argument('x_target_lon', help='Initial point longitude in degrees')
    parser_real.add_argument('x_target_lat', help='Initial point latitude in degrees')
    parser_real.add_argument('start_date', help='Start date')
    parser_real.add_argument('airspeed', help='Airspeed in meters per seconds')
    parser_real.add_argument('altitude', help='Pressure level in hPa')
    parser_real.add_argument('--rft', help='Perform front tracking', action='store_const', const=True,
                             default=False)

    args = parser.parse_args(sys.argv[1:])
    if 'test_id' in args.__dict__:
        test_dir = os.path.dirname(__file__)
        pb_id = args.test_id
        try:
            _pb_id = int(pb_id)
        except ValueError:
            if type(pb_id) == str:
                try:
                    _pb_id = list(map(lambda x: x[1], IndexedProblem.problems)).index(pb_id)
                except ValueError:
                    print(f'Unknown problem "{pb_id}"')
                    print(f'Available problems : {sorted(list(map(lambda x: x[1], IndexedProblem.problems)))}')
                    exit(1)
            else:
                print(f'Please provide problem number (int) or problem name (string)')
                exit(1)
        pb_name = IndexedProblem.problems[_pb_id][1]
        ddf = DDFmanager()
        ddf.setup()
        ddf.set_case(f'{pb_name}')
        ddf.clean_output_dir()
        pb = IndexedProblem(_pb_id)
        t_upper_bound = pb.time_scale if pb.time_scale is not None else pb.l_ref / pb.model.v_a
        solver_ef = SolverEF(pb, t_upper_bound, max_steps=700, rel_nb_ceil=0.02, quick_solve=1)
        res = solver_ef.solve(verbose=2)
        ddf.dump_wind(pb.model.wind, nx=101, ny=101, nt=10, bl=pb.bl, tr=pb.tr, coords=pb.coords)
        extremals = solver_ef.get_trajs()
        ddf.dump_trajs(extremals)
        ddf.dump_trajs([res.traj])
        ddf.ps.load_from_problem(pb)
        ddf.ps.dump()
        print(f'Results saved to {ddf.case_dir}')

    else:
        x_init = np.array([Utils.to_m180_180(float(args.x_init_lon)), float(args.x_init_lat)])
        x_target = np.array([Utils.to_m180_180(float(args.x_target_lon)), float(args.x_target_lat)])
        airspeed = float(args.airspeed)
        print('WARNING : Altitude not read yet. Always assumed at 1000hPa.')

        duration = 2 * Utils.distance(Utils.DEG_TO_RAD * x_init, Utils.DEG_TO_RAD * x_target,
                                      coords=Utils.COORD_GCS) / airspeed

        st_d = args.start_date
        year = int(st_d[:4])
        aargs = [int(st_d[4 + 2 * i:4 + 2 * (i + 1)]) for i in range(4)]
        start_date = datetime(year, *aargs)
        stop_date = datetime.fromtimestamp(start_date.timestamp() + duration)
        ddf = DDFmanager()
        ddf.setup()

        ddf.retrieve_wind(start_date, stop_date, level='1000', res='0.5')
        case_name = ddf.format_cname(x_init, x_target, start_date.timestamp())

        cache_wind = False
        cache_rff = False

        # This instance prints absolute elapsed time between operations
        chrono = Chrono()

        # Create a file manager to dump problem data
        mdfm = DDFmanager(cache_wind=cache_wind, cache_rff=cache_rff)
        mdfm.setup()
        case_name = f'zermelo_{case_name}'
        mdfm.set_case(case_name)
        mdfm.clean_output_dir()

        # Space and time discretization
        # Will be used to save wind when wind is analytical and shall be sampled
        # Will also be used by front tracking module
        nx_rft = 101
        ny_rft = 101
        nt_rft = 20

        pb = DatabaseProblem(x_init=Utils.DEG_TO_RAD * x_init,
                             x_target=Utils.DEG_TO_RAD * x_target, airspeed=airspeed,
                             t_start=start_date.timestamp(), t_end=stop_date.timestamp(),
                             altitude=args.altitude,
                             resolution='0.5')

        # pb.flatten()

        if not cache_wind:
            chrono.start('Dumping windfield to file')
            mdfm.dump_wind(pb.model.wind, nx=nx_rft, ny=ny_rft, nt=nt_rft, bl=pb.bl, tr=pb.tr)
            chrono.stop()

        # Setting the extremal solver
        solver_ef = solver = SolverEF(pb, pb.time_scale, max_steps=700, rel_nb_ceil=0.02, quick_solve=True)

        chrono.start('Solving problem using extremal field (EF)')
        res_ef = solver_ef.solve()
        chrono.stop()
        if res_ef.status:
            # Solution found
            # Save optimal trajectory
            mdfm.dump_trajs([res_ef.traj])
            print(f'Target reached in : {Utils.time_fmt(res_ef.duration)}')
        else:
            print('No solution found')

        # Save extremal field for display purposes
        extremals = solver_ef.get_trajs()
        mdfm.dump_trajs(extremals)
        mdfm.dump_obs(pb, nx_rft, ny_rft)

        pb.orthodromic()
        mdfm.dump_trajs([pb.trajs[-1]])

        # pb.flatten()

        # Setting the front tracking solver
        if args.rft:
            solver_rp = SolverRP(pb, nx_rft, ny_rft, nt_rft)
            if cache_rff:
                solver_rp.rft.load_cache(os.path.join(mdfm.case_dir, 'rff.h5'))

            chrono.start('Solving problem using reachability front tracking (RFT)')
            res_rp = solver_rp.solve()
            chrono.stop()
            if res_rp.status:
                # Save optimal trajectory
                mdfm.dump_trajs([res_rp.traj])
                print(f'Target reached in : {Utils.time_fmt(res_rp.duration)}')

            # Save fronts for display purposes
            if not cache_rff:
                solver_rp.rft.dump_rff(mdfm.case_dir)

        # Extract information for display and write it to output
        mdfm.ps.load_from_problem(pb)
        mdfm.ps.dump()
        # Also copy the script that produced the result to output dir for later reproduction
        mdfm.save_script(__file__)

        # mdfm.set_case('example_dakar-natal-constr*')
        # mdfm.dump_trajs([res_ef.traj])

        print(f'Results saved to {mdfm.case_dir}')
