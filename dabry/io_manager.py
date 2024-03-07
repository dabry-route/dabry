import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from time import strftime
from typing import Optional, List

import h5py
import numpy as np
import pygrib
from numpy import ndarray

from dabry.flowfield import FlowField, save_ff
from dabry.misc import Utils, Units, Coords
from dabry.penalty import DiscretePenalty
from dabry.trajectory import Trajectory

"""
io_manager.py
Handles the writing and reading of special data format for trajectories, 
flow field, reachability functions and obstacles.

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


class IOManager:
    """
    This class handles the writing and reading of files from or to disk
    """

    def __init__(self, name: str, case_dir: Optional[str] = None, cache_ff=False, cache_rff=False):
        self.obs_filename = 'obs.h5'
        self.pen_filename = 'penalty.h5'
        self.cache_ff = cache_ff
        self.cache_rff = cache_rff
        self._dabry_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.case_dir = case_dir if case_dir is not None else os.path.join(self._dabry_root_dir, 'output', name)
        self.case_name = name

    @property
    def output_dir(self):
        return os.path.dirname(self.case_dir)

    def set_case_dir(self, dirpath: str):
        self.case_dir = os.path.abspath(dirpath)

    def setup_dir(self):
        if not os.path.exists(self.case_dir):
            os.mkdir(self.case_dir)

    @property
    def _cds_ff_db_dir(self):
        return os.path.join(self._dabry_root_dir, 'data', 'cds')

    @property
    def trajs_dir(self):
        return os.path.join(self.case_dir, 'trajs')

    @property
    def ff_fpath(self):
        return os.path.join(self.case_dir, 'ff.npz')

    @property
    def pb_data_fpath(self):
        return os.path.join(self.case_dir, self.case_name + '.json')

    @property
    def coords(self) -> Coords:
        if not os.path.exists(self.ff_fpath):
            raise FileNotFoundError('Flow field file not found "%s"' % self.ff_fpath)
        return Coords.from_string(np.load(self.ff_fpath, mmap_mode='r')['coords'])

    def border(self, name: str):
        if not os.path.exists(self.pb_data_fpath):
            return np.load(self.ff_fpath)['bounds'].transpose()[0 if name == 'bl' else 1][-2:]
            # raise FileNotFoundError('Problem data file not found "%s"' % self.pb_data_fpath)
        return np.array(json.load(open(self.pb_data_fpath))[name])

    @property
    def bl(self) -> ndarray:
        return self.border('bl')

    @property
    def tr(self) -> ndarray:
        return self.border('tr')

    @property
    def x_init(self) -> ndarray:
        return np.array(json.load(open(self.pb_data_fpath))["x_init"])

    @property
    def x_target(self) -> ndarray:
        return np.array(json.load(open(self.pb_data_fpath))["x_target"])

    @property
    def target_radius(self) -> ndarray:
        return np.array(json.load(open(self.pb_data_fpath))["target_radius"])

    def setup_trajs(self):
        self.setup_dir()
        if not os.path.exists(self.trajs_dir):
            os.mkdir(self.trajs_dir)

    def clean_output_dir(self):
        if not os.path.exists(self.case_dir):
            return
        for filename in os.listdir(self.case_dir):
            if filename.endswith('ff.npz') and self.cache_ff:
                continue
            if filename.endswith('rff.h5') and self.cache_rff:
                continue
            path = os.path.join(self.case_dir, filename)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    def save_script(self, script_path):
        shutil.copy(script_path, os.path.join(self.case_dir, 'script.py'))

    def save_traj(self, traj: Trajectory, name: str, target_dir: Optional[str] = None):
        if target_dir is None:
            target_dir = self.trajs_dir
        traj.save(name, target_dir)

    def save_trajs(self, trajs: List[Trajectory], group_name: Optional[str] = None):
        if len(trajs) == 0:
            return
        self.setup_trajs()
        if group_name is not None:
            target_dir = os.path.join(self.trajs_dir, group_name)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
        else:
            target_dir = self.trajs_dir
        for i_traj, traj in enumerate(trajs):
            name = str(i_traj).rjust(1 + int(np.log10(len(trajs) - 1)), '0')
            self.save_traj(traj, name, target_dir)

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

    def save_ff(self, ff: FlowField,
                nx: Optional[int] = None,
                ny: Optional[int] = None,
                nt: Optional[int] = None,
                bl: Optional[ndarray] = None,
                tr: Optional[ndarray] = None):
        if os.path.exists(self.ff_fpath) and self.cache_ff:
            return
        self.setup_dir()
        save_ff(ff, self.ff_fpath, nx=nx, ny=ny, nt=nt, bl=bl, tr=tr)

    def dump_penalty(self, penalty: DiscretePenalty):
        filepath = os.path.join(self.case_dir, self.pen_filename)
        with h5py.File(filepath, 'w') as f:
            f.attrs['coords'] = Coords.GCS.value
            f.attrs['units_grid'] = Units.RADIANS.value
            dset = f.create_dataset('data', penalty.data.shape, dtype='f8')
            dset[:] = penalty.data
            dset = f.create_dataset('ts', penalty.ts.shape, dtype='f8')
            dset[:] = penalty.ts
            dset = f.create_dataset('grid', penalty.grid.shape, dtype='f8')
            dset[:] = penalty.grid

    def dump_ff_from_grib2(self, srcfiles, bl, tr, dstname=None, coords=Coords.GCS):
        if coords == Coords.CARTESIAN:
            print('Cartesian conversion not handled yet', file=sys.stderr)
            exit(1)
        if type(srcfiles) == str:
            srcfiles = [srcfiles]
        filename = self.ff_filename if dstname is None else dstname
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
            dates[k] = IOManager.grib_date_to_unix(os.path.basename(grbfile))

        with h5py.File(filepath, 'w') as f:
            f.attrs['coords'] = coords.value
            f.attrs['units_grid'] = Units.RADIANS.value
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
        try:
            import pyproj
        except ImportError:
            raise ImportError('"pyproj" module required for geometrical grib2 file selection')
        # data_dir = '/home/bastien/Documents/data/wind/ncdc/20210929'
        # output_path = '/home/bastien/Documents/data/wind/ncdc/'
        x_init = np.array(x_init)
        x_target = np.array(x_target)
        middle = Utils.middle(Utils.DEG_TO_RAD * x_init, Utils.DEG_TO_RAD * x_target, Coords.GCS)
        distance = Utils.distance(Utils.DEG_TO_RAD * x_init, Utils.DEG_TO_RAD * x_target, Coords.GCS)
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
        print(self.dump_ff_from_grib2(grib_fps, bl, tr))

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
        days_required = IOManager.days_between(start_date, stop_date)
        db_path = os.path.join(output_dir, resolution, pressure_level)
        print(output_dir)
        print(db_path)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        for ff_file in os.listdir(db_path):
            wf_date = ff_file.split('.')[0]
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
            try:
                import cdsapi
            except ImportError:
                raise ImportError('"cdsapi" module required to query ERA5')
            server = cdsapi.Client()
            kwargs = {
                "variable": ['u_component_of_wind', 'v_component_of_wind'],
                "pressure_level": pressure_level,
                "product_type": "reanalysis",
                "year": day_required[:4],
                "month": day_required[4:6],
                "day": day_required[6:8],
                'time': ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
                "grid": ['0.5', '0.5'],
                "format": "grib"
            }

            ff_name = f'{day_required}.grb2'
            server.retrieve("reanalysis-era5-pressure-levels", kwargs, os.path.join(db_path, ff_name))
