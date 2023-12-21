import os
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from time import strftime
from typing import Optional, List

import cdsapi
import h5py
import numpy as np
import pygrib
import pyproj
from numpy import ndarray

from dabry.misc import Utils
from dabry.penalty import DiscretePenalty
from dabry.trajectory import Trajectory
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
        self.module_dir: Optional[str] = None
        self.cds_wind_db_dir: Optional[str] = None
        self.case_dir: Optional[str] = None
        self.trajs_filename = 'trajectories.h5'
        self.wind_filename = 'wind'
        self.obs_filename = 'obs.h5'
        self.pen_filename = 'penalty.h5'
        self.case_name: Optional[str] = None
        self.cache_wind = cache_wind
        self.cache_rff = cache_rff

    def setup(self, module_dir: Optional[str] = None):
        if module_dir is not None:
            self.module_dir = module_dir
        else:
            path = os.environ.get('DABRYPATH')
            if path is None:
                path = '..'
            self.module_dir = path
        self.cds_wind_db_dir = os.path.join(self.module_dir, 'data', 'cds')

    def set_case(self, case_name, module_dir: Optional[str] = None):
        self.setup(module_dir=module_dir)
        self.case_name = case_name.split('/')[-1]
        if self.module_dir is None:
            raise Exception('Output directory not specified yet')
        self.case_dir = os.path.join(self.module_dir, 'output', self.case_name)
        if not os.path.exists(self.case_dir):
            os.mkdir(self.case_dir)

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

    def dump_trajs(self, trajs: List[Trajectory], filename=None):
        filename = self.trajs_filename if filename is None else filename
        filepath = os.path.join(self.case_dir, filename)
        with h5py.File(filepath, "a") as f:
            index = 0
            if len(f.keys()) != 0:
                index = int(max(f.keys(), key=int)) + 1
            for i, traj in enumerate(trajs):
                trajgroup = f.create_group(str(index + i))
                if traj.info_dict is not None:
                    for k, v in traj.info_dict.items():
                        trajgroup.attrs[k] = v
                trajgroup.attrs['coords'] = traj.coords
                n = traj.times.shape[0]
                dset = trajgroup.create_dataset('data', traj.states.shape, dtype='f8')
                dset[:] = traj.states

                dset = trajgroup.create_dataset('ts', traj.times.shape, dtype='f8')
                dset[:] = traj.times

                if traj.controls is not None:
                    dset = trajgroup.create_dataset('controls', traj.controls.shape, dtype='f8')
                    dset[:] = traj.controls

                if traj.costates is not None:
                    dset = trajgroup.create_dataset('adjoints', traj.costates.shape, dtype='f8')
                    dset[:] = traj.costates

                if traj.cost is not None:
                    dset = trajgroup.create_dataset('cost', n, dtype='f8')
                    dset[:] = traj.cost

    # def dump_obs(self, nx=100, ny=100):
    #     filepath = os.path.join(self.case_dir, self.obs_filename)
    #     with h5py.File(os.path.join(filepath), 'w') as f:
    #         f.attrs['coords'] = pb.coords
    #         delta_x = (pb.tr[0] - pb.bl[0]) / (nx - 1)
    #         delta_y = (pb.tr[1] - pb.bl[1]) / (ny - 1)
    #         dset = f.create_dataset('grid', (nx, ny, 2), dtype='f8')
    #         X, Y = np.meshgrid(pb.bl[0] + delta_x * np.arange(nx),
    #                            pb.bl[1] + delta_y * np.arange(ny), indexing='ij')
    #         dset[:, :, 0] = X
    #         dset[:, :, 1] = Y
    #
    #         obs_val = np.infty * np.ones((nx, ny))
    #         obs_id = -1 * np.ones((nx, ny))
    #         for k, obs in enumerate(pb.obstacles):
    #             for i in range(nx):
    #                 for j in range(ny):
    #                     point = np.array((pb.bl[0] + delta_x * i, pb.bl[1] + delta_y * j))
    #                     val = obs.value(point)
    #                     if val < 0. and val < obs_val[i, j]:
    #                         obs_val[i, j] = val
    #                         obs_id[i, j] = k
    #         dset = f.create_dataset('data', (nx, ny), dtype='f8')
    #         dset[:, :] = obs_id

    @staticmethod
    def grib_date_to_unix(grib_filename):
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

    def dump_wind(self, ff: Wind, nx: Optional[int] = None, ny: Optional[int] = None, nt: Optional[int] = None,
                  bl: Optional[ndarray] = None, tr: Optional[ndarray] = None):
        filepath = os.path.join(self.case_dir, self.wind_filename)
        if os.path.exists(filepath) and self.cache_wind:
            return
        DDFmanager.dump_wind_to_file(ff, filepath, nx=nx, ny=ny, nt=nt, bl=bl, tr=tr, fmt='h5')

    @staticmethod
    def _cast_to_discrete_wind(ff: Wind, nx: Optional[int] = None, ny: Optional[int] = None, nt: Optional[int] = None,
                               bl: Optional[ndarray] = None, tr: Optional[ndarray] = None):
        if not isinstance(ff, DiscreteWind):
            nx = 50 if nx is None else nx
            ny = 50 if ny is None else ny
            nt = 25 if nt is None else nt
            if bl is None or tr is None:
                raise Exception('Missing bounding box (bl, tr) to sample analytical flow field')
            return DiscreteWind.from_wind(ff, np.array((bl, tr)).transpose(), nx=nx, ny=ny, nt=nt, force_no_diff=True)
        else:
            if nx is not None or ny is not None or nt is not None:
                warnings.warn('Grid shape (nt, nx, ny) provided but resampling of DiscreteWind not implemented yet. '
                              'Continuing with wind native grid')
            return ff

    @classmethod
    def dump_wind_to_file(cls, ff: Wind, filepath: str, fmt='npz',
                          nx: Optional[int] = None, ny: Optional[int] = None, nt: Optional[int] = None,
                          bl: Optional[ndarray] = None, tr: Optional[ndarray] = None):
        if fmt not in ['h5', 'npz']:
            raise Exception(f'Unknown output format "{fmt}"')
        dff = DDFmanager._cast_to_discrete_wind(ff, nx, ny, nt, bl, tr)
        if fmt == 'h5':
            with h5py.File(filepath + '.' + fmt, 'w') as f:
                f.attrs['coords'] = dff.coords
                f.attrs['units_grid'] = Utils.U_METERS if dff.coords == Utils.COORD_CARTESIAN else Utils.U_RAD
                if dff.values.ndim == 4:
                    dset = f.create_dataset('data', dff.values.shape, dtype='f8')
                    dset[:] = dff.values
                else:
                    dset = f.create_dataset('data', (1,) + dff.values.shape, dtype='f8')
                    dset[0, :] = dff.values
                if dff.values.ndim == 4:
                    dset = f.create_dataset('ts', (dff.values.shape[0],), dtype='f8')
                    dset[:] = np.linspace(dff.bounds[0, 0], dff.bounds[0, 1], dff.values.shape[0])
                else:
                    dset = f.create_dataset('ts', (1,), dtype='f8')
                    dset[:] = dff.t_start
                _nx, _ny = dff.values.shape[:2] if dff.values.ndim == 3 else dff.values.shape[1:3]
                dset = f.create_dataset('grid', (_nx, _ny, 2), dtype='f8')
                dset[:] = np.stack(np.meshgrid(np.linspace(dff.bounds[-2, 0], dff.bounds[-2, 1], _nx),
                                               np.linspace(dff.bounds[-1, 0], dff.bounds[-1, 1], _ny), indexing='ij'), -1)
        else:
            # fmt == 'npz'
            np.savez(filepath + '.' + fmt, values=dff.values, bounds=dff.bounds, coords=np.array(dff.coords))

    def dump_penalty(self, penalty: DiscretePenalty):
        filepath = os.path.join(self.case_dir, self.pen_filename)
        with h5py.File(filepath, 'w') as f:
            f.attrs['coords'] = 'gcs'
            f.attrs['units_grid'] = Utils.U_RAD
            dset = f.create_dataset('data', penalty.data.shape, dtype='f8')
            dset[:] = penalty.data
            dset = f.create_dataset('ts', penalty.ts.shape, dtype='f8')
            dset[:] = penalty.ts
            dset = f.create_dataset('grid', penalty.grid.shape, dtype='f8')
            dset[:] = penalty.grid

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
            dates[k] = DDFmanager.grib_date_to_unix(os.path.basename(grbfile))

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

    @staticmethod
    def format_cname(x_init, x_target, t_start):
        """
        Standard case name
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

    @staticmethod
    def days_between(start_date, stop_date):
        l = []
        day_count = (stop_date - start_date).days + 1
        for single_date in [d for d in (start_date + timedelta(n) for n in range(day_count)) if d <= stop_date]:
            l.append(strftime("%Y%m%d", single_date.timetuple()))
        return l

    @staticmethod
    def query_era5(start_date: datetime, stop_date: datetime,
                   output_dir: str, pressure_level='1000', resolution='0.5'):
        in_cache = []
        days_required = DDFmanager.days_between(start_date, stop_date)
        db_path = os.path.join(output_dir, resolution, pressure_level)
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
                "variable": ['u_component_of_wind', 'v_component_of_wind', 'specific_rain_water_content'],
                "pressure_level": pressure_level,
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
