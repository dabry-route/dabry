import os
import shutil
import sys
from datetime import datetime, timedelta
from time import strftime

import cdsapi
import h5py
import numpy as np
import pygrib
import pyproj

from dabry.misc import Utils
from dabry.params_summary import ParamsSummary
from dabry.problem import NavigationProblem
from dabry.wind import Wind, DiscreteWind

"""
ddf_manager.py
Handles the writing and reading of special data format for trajectories, 
wind, reachability functions and obstacles.

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


class DDFmanager:
    """
    This class handles the writing and reading of Dabry Data Format (DDF) files
    """

    def __init__(self, cache_wind=False, cache_rff=False):
        self.module_dir = None
        self.cds_wind_db_dir = None
        self.case_dir = None
        self.trajs_filename = 'trajectories.h5'
        self.wind_filename = 'wind.h5'
        self.obs_filename = 'obs.h5'
        self.case_name = None
        self.ps = ParamsSummary()
        self.cache_wind = cache_wind
        self.cache_rff = cache_rff

    def setup(self, module_dir=None):
        if module_dir is not None:
            self.module_dir = module_dir
        else:
            path = os.environ.get('DABRYPATH')
            if path is None:
                raise Exception('Unable to set output dir automatically. Please set DABRYPATH variable.')
            self.module_dir = path
        self.cds_wind_db_dir = os.path.join(self.module_dir, 'data', 'cds')

    def set_case(self, case_name):
        self.case_name = case_name.split('/')[-1]
        if self.module_dir is None:
            raise Exception('Output directory not specified yet')
        self.case_dir = os.path.join(self.module_dir, 'output', self.case_name)
        if not os.path.exists(self.case_dir):
            os.mkdir(self.case_dir)
        self.ps.setup(self.case_dir)

    def clean_output_dir(self):
        if not os.path.exists(self.case_dir):
            return
        for filename in os.listdir(self.case_dir):
            if filename.endswith('wind.h5') and self.cache_wind:
                continue
            if filename.endswith('rff.h5') and self.cache_rff:
                continue
            path = os.path.join(self.case_dir, filename)
            if os.path.isdir(path):
                for file in os.listdir(path):
                    os.remove(os.path.join(path, file))
                os.rmdir(path)
            else:
                os.remove(path)

    def save_script(self, script_path):
        shutil.copy(script_path, self.case_dir)

    def dump_trajs(self, traj_list, filename=None, no_relabel=False):
        filename = self.trajs_filename if filename is None else filename
        filepath = os.path.join(self.case_dir, filename)
        with h5py.File(filepath, "a") as f:
            index = 0
            if len(f.keys()) != 0:
                index = int(max(f.keys(), key=int)) + 1
            for i, traj in enumerate(traj_list):
                trajgroup = f.create_group(str(index + i))
                trajgroup.attrs['type'] = traj.type
                trajgroup.attrs['coords'] = traj.coords
                trajgroup.attrs['interrupted'] = traj.interrupted
                trajgroup.attrs['last_index'] = traj.last_index
                if not no_relabel:
                    trajgroup.attrs['label'] = index + i
                else:
                    trajgroup.attrs['label'] = traj.label
                trajgroup.attrs['info'] = traj.info
                n = traj.last_index + 1
                dset = trajgroup.create_dataset('data', (n, 2), dtype='f8')
                dset[:, :] = traj.points[:n]

                dset = trajgroup.create_dataset('ts', (n,), dtype='f8')
                dset[:] = traj.timestamps[:n]

                dset = trajgroup.create_dataset('controls', (n,), dtype='f8')
                dset[:] = traj.controls[:n]

                if hasattr(traj, 'adjoints'):
                    dset = trajgroup.create_dataset('adjoints', (n, 2), dtype='f8', fillvalue=0.)
                    dset[:, :] = traj.adjoints[:n]

                if hasattr(traj, 'transver'):
                    dset = trajgroup.create_dataset('transver', n, dtype='f8', fillvalue=0.)
                    dset[:] = traj.transver[:n]

                if hasattr(traj, 'airspeed'):
                    dset = trajgroup.create_dataset('airspeed', n, dtype='f8', fillvalue=0.)
                    dset[:] = traj.airspeed[:n]

                if hasattr(traj, 'energy'):
                    dset = trajgroup.create_dataset('energy', n - 1, dtype='f8', fillvalue=0.)
                    dset[:] = traj.energy[:n - 1]

    def dump_obs(self, pb: NavigationProblem, nx=100, ny=100):
        filepath = os.path.join(self.case_dir, self.obs_filename)
        with h5py.File(os.path.join(filepath), 'w') as f:
            f.attrs['coords'] = pb.coords
            delta_x = (pb.tr[0] - pb.bl[0]) / (nx - 1)
            delta_y = (pb.tr[1] - pb.bl[1]) / (ny - 1)
            dset = f.create_dataset('grid', (nx, ny, 2), dtype='f8')
            X, Y = np.meshgrid(pb.bl[0] + delta_x * np.arange(nx),
                               pb.bl[1] + delta_y * np.arange(ny), indexing='ij')
            dset[:, :, 0] = X
            dset[:, :, 1] = Y

            obs_val = np.infty * np.ones((nx, ny))
            obs_id = -1 * np.ones((nx, ny))
            for k, obs in enumerate(pb.obstacles):
                for i in range(nx):
                    for j in range(ny):
                        point = np.array((pb.bl[0] + delta_x * i, pb.bl[1] + delta_y * j))
                        val = obs.value(point)
                        if val < 0. and val < obs_val[i, j]:
                            obs_val[i, j] = val
                            obs_id[i, j] = k
            dset = f.create_dataset('data', (nx, ny), dtype='f8')
            dset[:, :] = obs_id

    def _grib_date_to_unix(self, grib_filename):
        date, hm = grib_filename.split('_')[2:4]
        hours = int(hm[:2])
        minutes = int(hm[2:])
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:])
        dt = datetime(year, month, day, hours, minutes)
        return dt.timestamp()

    def print_trajs(self, filepath):
        with h5py.File(filepath, 'r') as f:
            for traj in f.values():
                print(traj)
                for attr, val in traj.attrs.items():
                    print(f'{attr} : {val}')

    def dump_wind(self, wind: Wind, filename=None, nx=None, ny=None, nt=None, bl=None, tr=None,
                  coords=Utils.COORD_CARTESIAN,
                  force_analytical=False):
        if wind.is_dumpable == 0:
            print('Error : Wind is not dumpable to file', file=sys.stderr)
            exit(1)
        filename = self.wind_filename if filename is None else filename
        filepath = os.path.join(self.case_dir, filename)
        if os.path.exists(filepath) and self.cache_wind:
            return
        if wind.is_dumpable == 1:
            if nx is None or ny is None:
                print(f'Please provide grid shape "nx=..., ny=..." to sample analytical wind "{wind}"')
                exit(1)
            dwind = DiscreteWind(force_analytical=force_analytical)
            kwargs = {}
            if nt is not None:
                kwargs['nt'] = nt
            dwind.load_from_wind(wind, nx, ny, bl, tr, coords, nodiff=True, **kwargs)
        else:
            dwind = wind
        with h5py.File(filepath, 'w') as f:
            f.attrs['coords'] = dwind.coords
            f.attrs['units_grid'] = Utils.U_METERS if dwind.coords == Utils.COORD_CARTESIAN else Utils.U_RAD
            f.attrs['analytical'] = dwind.is_analytical
            if dwind.lon_0 is not None:
                f.attrs['lon_0'] = dwind.lon_0
            if dwind.lat_0 is not None:
                f.attrs['lat_0'] = dwind.lat_0
            f.attrs['unstructured'] = dwind.unstructured
            dset = f.create_dataset('data', (dwind.nt, dwind.nx, dwind.ny, 2), dtype='f8')
            dset[:] = dwind.uv
            dset = f.create_dataset('ts', (dwind.nt,), dtype='f8')
            dset[:] = dwind.ts
            dset = f.create_dataset('grid', (dwind.nx, dwind.ny, 2), dtype='f8')
            dset[:] = dwind.grid

    def dump_wind_from_grib2(self, srcfiles, bl, tr, dstname=None, coords=Utils.COORD_GCS):
        if coords == Utils.COORD_CARTESIAN:
            print('Cartesian conversion not handled yet', file=sys.stderr)
            exit(1)
        if type(srcfiles) == str:
            srcfiles = [srcfiles]
        filename = self.wind_filename if dstname is None else dstname
        filepath = os.path.join(self.case_dir, filename)

        def process(grbfile, setup=False, nx=None, ny=None):
            grbs = pygrib.open(grbfile)
            grb = grbs.select(name='U component of wind', typeOfLevel='isobaricInhPa', level=1000)[0]
            lon_b = (bl[0], tr[0])  # Utils.rectify(bl[0], tr[0])
            U, lats, lons = grb.data(lat1=bl[1], lat2=tr[1], lon1=lon_b[0], lon2=lon_b[1])
            grb = grbs.select(name='V component of wind', typeOfLevel='isobaricInhPa', level=1000)[0]
            V, _, _ = grb.data(lat1=bl[1], lat2=tr[1], lon1=lon_b[0], lon2=lon_b[1])
            if setup:
                if lons.max() > 180.:
                    newlons = np.zeros(lons.shape)
                    newlons[:] = lons - 360.
                    lons[:] = newlons

                newlats = np.zeros(lats.shape)
                newlats[:] = lats[::-1, :]
                lats[:] = newlats

                ny, nx = U.shape
                grid = np.zeros((nx, ny, 2))
                grid[:] = np.transpose(np.stack((lons, lats)), (2, 1, 0))
                return nx, ny, grid
            else:
                if nx is None or ny is None:
                    print('Missing nx or ny', file=sys.stderr)
                    exit(1)
                UV = np.zeros((nx, ny, 2))
                UV[:] = np.transpose(np.stack((U, V)), (2, 1, 0))
                UV[:] = UV[:, ::-1, :]
                return UV

        # First fetch grid parameters
        nx, ny, grid = process(srcfiles[0], setup=True)
        nt = len(srcfiles)
        UVs = np.zeros((nt, nx, ny, 2))
        dates = np.zeros((nt,))
        for k, grbfile in enumerate(srcfiles):
            UV = process(grbfile, nx=nx, ny=ny)
            UVs[k, :] = UV
            dates[k] = self._grib_date_to_unix(os.path.basename(grbfile))

        with h5py.File(filepath, 'w') as f:
            f.attrs['coords'] = coords
            f.attrs['units_grid'] = Utils.U_RAD
            f.attrs['analytical'] = False
            dset = f.create_dataset('data', (nt, nx, ny, 2), dtype='f8')
            dset[:] = UVs
            dset = f.create_dataset('ts', (nt,), dtype='f8')
            dset[:] = dates
            dset = f.create_dataset('grid', (nx, ny, 2), dtype='f8')
            dset[:] = Utils.DEG_TO_RAD * grid

        return nt, nx, ny

    def format_cname(self, x_init, x_target, t_start):
        """
        Stardand case name
        """
        slon = str(abs(round(x_init[0]))) + ('W' if x_init[0] < 0 else 'E')
        slat = str(abs(round(x_init[1]))) + ('S' if x_init[1] < 0 else 'N')
        tlon = str(abs(round(x_target[0]))) + ('W' if x_target[0] < 0 else 'E')
        tlat = str(abs(round(x_target[1]))) + ('S' if x_target[1] < 0 else 'N')
        start_date = datetime.fromtimestamp(t_start)
        return f'{slon}_{slat}_{tlon}_{tlat}_{strftime("%Y%m%d", start_date.timetuple())}_{start_date.hour:0>2}'

    def process_grib(self, input_gribs, output_path, x_init, x_target, print_name=False):
        """
        :param x_init: Initial point (lon, lat) in degrees
        :param x_target: Target point (lon, lat) in degrees
        """
        # data_dir = '/home/bastien/Documents/data/wind/ncdc/20210929'
        # output_path = '/home/bastien/Documents/data/wind/ncdc/'
        x_init = np.array(x_init)
        x_target = np.array(x_target)
        middle = Utils.middle(Utils.DEG_TO_RAD * x_init, Utils.DEG_TO_RAD * x_target, Utils.COORD_GCS)
        distance = Utils.distance(Utils.DEG_TO_RAD * x_init, Utils.DEG_TO_RAD * x_target, Utils.COORD_GCS)
        factor = 1.2
        lon_0, lat_0 = Utils.RAD_TO_DEG * middle[0], Utils.RAD_TO_DEG * middle[1]
        proj = pyproj.Proj(proj='ortho', lon_0=lon_0, lat_0=lat_0)

        bl = np.array(proj(*middle)) - 0.5 * factor * distance * np.ones(2)
        tr = np.array(proj(*middle)) + 0.5 * factor * distance * np.ones(2)
        bl = 1. * (np.array((lon_0, lat_0)) + np.array(proj(*bl, inverse=True)))
        tr = 1. * (np.array((lon_0, lat_0)) + np.array(proj(*tr, inverse=True)))

        # Data files
        gribfiles = input_gribs
        date_start = gribfiles[0].split('_')[2]
        hour_start = gribfiles[0].split('_')[3][:2]

        slon = str(abs(round(bl[0]))) + ('W' if bl[0] < 0 else 'E')
        slat = str(abs(round(bl[1]))) + ('S' if bl[1] < 0 else 'N')
        tlon = str(abs(round(tr[0]))) + ('W' if tr[0] < 0 else 'E')
        tlat = str(abs(round(tr[1]))) + ('S' if tr[1] < 0 else 'N')

        name = f'{slon}_{slat}_{tlon}_{tlat}_{date_start}_{hour_start}'

        if print_name:
            return name

        output_dir = os.path.join(output_path, name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        data_dir = os.path.dirname(input_gribs[0])
        grib_fps = list(map(lambda gf: os.path.join(data_dir, gf), gribfiles))

        self.case_dir = output_dir
        print(self.dump_wind_from_grib2(grib_fps, bl, tr))

    def days_between(self, start_date, stop_date):
        l = []
        day_count = (stop_date - start_date).days + 1
        for single_date in [d for d in (start_date + timedelta(n) for n in range(day_count)) if d <= stop_date]:
            l.append(strftime("%Y%m%d", single_date.timetuple()))
        return l

    def retrieve_wind(self, start_date, stop_date, level='1000', res='0.5'):
        in_cache = []
        days_required = self.days_between(start_date, stop_date)
        db_path = os.path.join(self.cds_wind_db_dir, res, level)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        for wind_file in os.listdir(db_path):
            wf_date = wind_file.split('.')[0]
            if wf_date in days_required:
                in_cache.append(wf_date)
        for wf_date in in_cache:
            days_required.remove(wf_date)
        if len(days_required) == 0:
            sdstr = strftime('%Y%m%d %H:%M', start_date.timetuple())
            sdstr_stop = strftime('%Y%m%d %H:%M', stop_date.timetuple())
            print(f'Wind fully in cache ({sdstr} to {sdstr_stop})')
        else:
            print(f'Shall retrieve {len(days_required)} : {days_required}')

        for day_required in days_required:

            # server = ECMWFDataServer()
            server = cdsapi.Client()
            kwargs = {
                "variable": ['u_component_of_wind', 'v_component_of_wind'],
                "pressure_level": "1000",
                "product_type": "reanalysis",
                "year": day_required[:4],
                "month": day_required[4:6],
                "day": day_required[6:8],
                'time': [
                    '00:00',  # '01:00', '02:00',
                    '03:00',  # '04:00', '05:00',
                    '06:00',  # '07:00', '08:00',
                    '09:00',  # '10:00', '11:00',
                    '12:00',  # '13:00', '14:00',
                    '15:00',  # '16:00', '17:00',
                    '18:00',  # '19:00', '20:00',
                    '21:00',  # '22:00', '23:00',
                ],
                "grid": ['0.5', '0.5'],
                "format": "grib"
            }
            # kwargs = {
            #     'dataset': 'era20c',
            #     'stream': 'oper',
            #     'levtype': 'pl',
            #     'levelist': level,
            #     'param': '131.128/132.128',
            #     'step': '1',
            #     'type': 'an',
            #     'grid': f'{res}/{res}',
            # }
            # kwargs['date'] = day_required
            # kwargs['time'] = f'00/to/21/by/3'
            # kwargs['target'] = os.path.join(res_path, wind_name)

            wind_name = f'{day_required}.grb2'
            server.retrieve("reanalysis-era5-pressure-levels", kwargs, os.path.join(db_path, wind_name))
            # server.retrieve(kwargs)


if __name__ == '__main__':
    mdfm = DDFmanager()
    # mdfm.print_trajs('/home/bastien/Documents/work/dabry/output/example_front_tracking2/trajectories.h5')
    mdfm.setup('/home/bastien/Documents/work/test')
    data_dir = '/home/bastien/Documents/data/other'
    # data_filepath = os.path.join(data_dir, 'gfs_3_20090823_0600_000.grb2')
    data_filepath = os.path.join(data_dir, 'gfs_4_20220324_1200_000.grb2')
    bl = np.array((-50., -40.))
    tr = np.array((10., 40.))
    mdfm.dump_wind_from_grib2(data_filepath, bl, tr)
