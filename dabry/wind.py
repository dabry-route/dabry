import os
import sys
from abc import ABC
from datetime import datetime, timedelta
from math import exp, log
from typing import Optional, Union

import h5py
import numpy as np
import pygrib
from numpy import ndarray, pi, sin, cos
from tqdm import tqdm

from dabry.misc import Utils

"""
wind.py
Flow fields for navigation problems.

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


class Wind(ABC):

    def __init__(self, _lch=None, _rch=None, _op=None, t_start=0., t_end=None, nt_int=None):
        self._lch = _lch
        self._rch = _rch
        self._op = _op

        # When wind is time-varying, bounds for the time window
        # A none upper bound means no time variation
        self.t_start = t_start
        self.t_end = t_end

        # Intrisic number of time frames when analytically defined
        # flow fields depend on time-varying parameters
        self.nt_int = nt_int

        # True if dualization operation flips wind (mulitplication by -1)
        # False if dualization operation leaves wind unchanged
        self.dualizable = True

        self.coords = Utils.COORD_CARTESIAN

    def value(self, t, x):
        if self._lch is None:
            return self._value(t, x)
        if self._op == '+':
            return self._lch.value(t, x) + self._rch.value(t, x)
        if self._op == '-':
            return self._lch.value(t, x) - self._rch.value(t, x)
        if self._op == '*':
            if isinstance(self._lch, float):
                return self._lch + self._rch.value(t, x)
            if isinstance(self._rch, float):
                return self._lch.value(t, x) + self._rch
            raise Exception('Only scaling by float implemented for flow fields')

    def d_value(self, t, x):
        if self._lch is None:
            return self._d_value(t, x)
        if self._op == '+':
            return self._lch.d_value(t, x) + self._rch.d_value(t, x)
        if self._op == '-':
            return self._lch.d_value(t, x) - self._rch.d_value(t, x)
        if self._op == '*':
            if isinstance(self._lch, float):
                return self._lch + self._rch.d_value(t, x)
            if isinstance(self._rch, float):
                return self._lch.d_value(t, x) + self._rch
            raise Exception('Only scaling by float implemented for flow fields')

    def _value(self, t, x):
        pass

    def _d_value(self, t, x):
        pass

    def dualize(self):
        a = -1.
        if self.t_end is not None:
            mem = self.t_start
            self.t_start = self.t_end
            self.t_end = mem
        if not self.dualizable:
            a = 1.
        return a * self

    def __add__(self, other):
        """
        Add windfields
        :param other: Another windfield
        :return: The sum of the two windfields
        """
        if not isinstance(other, Wind):
            raise TypeError(f"Unsupported type for addition : {type(other)}")
        return Wind(_lch=self, _rch=other, _op='+')

    def __sub__(self, other):
        """
        Substracts windfields
        :param other: Another windfield
        :return: The substraction of the two windfields
        """
        if not isinstance(other, Wind):
            raise TypeError(f"Unsupported type for substraction : {type(other)}")
        return Wind(_lch=self, _rch=other, _op='-')

    def __mul__(self, other):
        """
        Handles the scaling of a windfield by a real number
        :param other: A real number (float)
        :return: The scaled windfield
        """
        if isinstance(other, int):
            other = float(other)
        if not isinstance(other, float):
            raise TypeError(f"Unsupported type for multiplication : {type(other)}")
        return Wind(_lch=self, _rch=other, _op='*', t_start=self.t_start, t_end=self.t_end)

    def __rmul__(self, other):
        return self.__mul__(other)

    def _index(self, t):
        """
        Get nearest lowest index for time discrete grid
        :param t: The required time
        :return: Nearest lowest index for time, coefficient for the linear interpolation
        """
        if self.t_end is None:
            return 0, 0.
        nt = self.nt_int
        if nt == 1:
            return 0, 0.
        # Bounds may not be in the right order if wind is dualized
        t_min, t_max = min(self.t_start, self.t_end), max(self.t_start, self.t_end)
        # Clip time to bounds
        if t <= t_min:
            return 0, 0.
        if t > t_max:
            return nt - 2, 1.
        tau = (t - t_min) / (t_max - t_min)
        i, alpha = int(tau * (nt - 1)), tau * (nt - 1) - int(tau * (nt - 1))
        if i == nt - 1:
            i = nt - 2
            alpha = 1.
        return i, alpha


class DiscreteWind(Wind):
    """
    Handles wind loading from H5 format and derivative computation
    """

    def __init__(self, values: ndarray, bounds: ndarray, coords: str, grad_values: Optional[ndarray] = None,
                 force_no_diff=False):
        super().__init__(nt_int=values.shape[0] if values.ndim == 4 else None)

        self.is_dumpable = 2

        if bounds.shape[0] != values.ndim - 1:
            raise Exception(f'Incompatible shape for values and bounds: '
                            f'values has {len(values.shape) - 1} + 1 dimensions '
                            f'so bounds must be of shape ({len(values.shape) - 1}, 2) ({bounds.shape} given)')

        self.values = values
        self.bounds = bounds
        self.coords = coords

        self.spacings = (bounds[:, 1] - bounds[:, 0]) / (np.array(self.values.shape[:-1]) -
                                                         np.ones(self.values.ndim - 1))

        self.grad_values = grad_values

        if not force_no_diff:
            if self.grad_values is None:
                self.compute_derivatives()

    @classmethod
    def from_h5(cls, filepath, **kwargs):
        """
        Loads wind data from H5 wind data
        :param filepath: The H5 file contaning wind data
        """

        # Fill the wind data array
        # print(f'{"Loading wind values...":<30}', end='')
        with h5py.File(filepath, 'r') as wind_data:
            coords = wind_data.attrs['coords']

            # Checking consistency before loading
            Utils.ensure_coords(coords)

            values = np.array(wind_data['data'])

            # Time bounds
            t_start = wind_data['ts'][0]
            t_end = None if wind_data['ts'].shape[0] == 1 else wind_data['ts'][-1]

            # Detecting millisecond-formated timestamps
            if np.any(np.array(wind_data['ts']) > 1e11):
                t_start /= 1000.
                if t_end is not None:
                    t_end /= 1000.
            f = Utils.DEG_TO_RAD if wind_data.attrs['units_grid'] == Utils.U_DEG else 1.

            bounds = np.stack((() if t_end is None else (np.array((t_start, t_end)),)) +
                              (f * wind_data['grid'][0, 0],
                               f * wind_data['grid'][-1, -1]), axis=0)

        return cls(values, bounds, coords, **kwargs)

    @classmethod
    def from_wind(cls, wind: Wind, grid_bounds, nx=100, ny=100, nt=50, **kwargs):
        t_start = wind.t_start
        t_end = wind.t_end

        bounds = np.stack((() if t_end is None else (np.array((t_start, t_end)),)) +
                          (grid_bounds[0], grid_bounds[1]), axis=0)
        shape = (() if t_end is None else (nt,)) + (nx, ny)
        spacings = (bounds[:, 1] - bounds[:, 0]) / (np.array(shape) - np.ones(bounds.shape[0]))
        _nt = nt if t_end is not None else 1
        values = np.zeros(shape + (2,))
        grad_values = np.zeros(shape + (2, 2)) if not kwargs.get('force_no_diff') else None
        for k in range(_nt):
            t = bounds[0, 0] + k * spacings[0] if t_end is not None else t_start
            for i in range(nx):
                for j in range(ny):
                    state = bounds[-2:, 0] + np.diag((i, j)) @ spacings[-2:]
                    if t_end is None:
                        values[i, j, :] = wind.value(t, state)
                    else:
                        values[k, i, j, :] = wind.value(t, state)
                    if not kwargs.get('force_no_diff'):
                        if t_end is None:
                            grad_values[i, j, ...] = wind.d_value(t, state)
                        else:
                            grad_values[k, i, j, ...] = wind.d_value(t, state)
        coords = Utils.COORD_GCS if isinstance(wind, DiscreteWind) and wind.coords == Utils.COORD_GCS \
            else Utils.COORD_CARTESIAN
        return cls(values, bounds, coords, grad_values=grad_values, **kwargs)

    @classmethod
    def from_cds(cls, grid_bounds: ndarray, t_start: Union[float | datetime], t_end: Union[float | datetime],
                 i_member: Optional[int] = None, resolution='0.5', pressure_level='1000',
                 data_path: Optional[str] = None):
        """
        :param grid_bounds: A (2,2) array where the zeroth element corresponds to the bounds of the zeroth axis
        and the first element to the bounds of the first axis
        :param t_start: The required start time for wind
        :param t_end: The required end time for wind
        :param i_member: The reanalysis ensemble member number
        :param resolution: The weather model grid resolution in degrees, e.g. '0.5'
        :param pressure_level: The pressure level in hPa, e.g. '1000', '500', '200'
        :param data_path: Force path to data to this value
        :return: A DiscreteWind corresponding to the query
        """
        if data_path is None:
            dirpath = os.path.join(os.environ.get('DABRYPATH'), 'data', 'cds', resolution, pressure_level)
        else:
            dirpath = os.path.join(data_path, resolution, pressure_level)
        t_start = t_start.timestamp() if isinstance(t_start, datetime) else t_start
        t_end = t_end.timestamp() if isinstance(t_end, datetime) else t_end
        single_frame = abs(t_start - t_end) < 60
        bl, tr = grid_bounds.transpose()[0], grid_bounds.transpose()[1]
        bl_d, tr_d = bl * Utils.RAD_TO_DEG, tr * Utils.RAD_TO_DEG

        lon_b = Utils.rectify(bl_d[0], tr_d[0])
        wind_file = sorted(os.listdir(dirpath))[0]
        wind_start_date = datetime(int(wind_file[:4]), int(wind_file[4:6]), int(wind_file[6:8]))
        wind_file = sorted(os.listdir(dirpath))[-1]
        wind_stop_date = datetime(int(wind_file[:4]), int(wind_file[4:6]), int(wind_file[6:8]))
        dates = [wind_start_date, wind_stop_date]
        for k, (t_bound, wind_date_bound, op) in enumerate(zip((t_start, t_end), (wind_start_date, wind_stop_date),
                                                            ('<', '>'))):
            adj = "lower" if op == "<" else "upper"
            if t_bound is None:
                print(f'[CDS wind] Using wind time {adj} bound {wind_date_bound}')
                dates[k] = wind_date_bound
            else:
                if not isinstance(t_bound, datetime):
                    # Assuming float
                    _date_bound = datetime.fromtimestamp(t_bound)
                else:
                    _date_bound = t_bound
                if (op == '<' and _date_bound < wind_date_bound) or (op == '>' and _date_bound > wind_date_bound):
                    print(f'[CDS wind] Clipping {adj} bound to wind time bound {wind_date_bound}')
                    dates[k] = wind_date_bound
                else:
                    dates[k] = _date_bound

        grb = pygrib.open(os.path.join(dirpath, os.listdir(dirpath)[0]))
        grb_u = grb.select(name='U component of wind')
        if lon_b[1] > 360:
            # Has to extract in two steps
            data1 = grb_u[0].data(lat1=bl_d[1], lat2=tr_d[1], lon1=lon_b[0], lon2=360)
            data2 = grb_u[0].data(lat1=bl_d[1], lat2=tr_d[1], lon1=0, lon2=lon_b[1] - 360)
            lons1 = Utils.DEG_TO_RAD * Utils.to_m180_180(data1[2]).transpose()
            lons2 = Utils.DEG_TO_RAD * Utils.to_m180_180(data2[2]).transpose()
            # lats1 = Utils.DEG_TO_RAD * data1[1].transpose()
            # lats2 = Utils.DEG_TO_RAD * data2[1].transpose()
            lons = np.concatenate((lons1, lons2))
            # lats = np.concatenate((lats1, lats2))
        else:
            data = grb_u[0].data(lat1=bl_d[1], lat2=tr_d[1], lon1=lon_b[0], lon2=lon_b[1])
            lons = Utils.DEG_TO_RAD * Utils.to_m180_180(data[2]).transpose()
            # lats = Utils.DEG_TO_RAD * data[1].transpose()

        shape = lons.shape

        # Get wind and timestamps
        uv_frames = []
        ts = []
        start_date_rounded = datetime(dates[0].year, dates[0].month, dates[0].day, 0, 0)
        stop_date_rounded = datetime(dates[1].year, dates[1].month, dates[1].day, 0, 0) + timedelta(days=1)
        startd_rounded = datetime(dates[0].year, dates[0].month, dates[0].day, 3 * (dates[0].hour // 3), 0)
        stopd_rounded = datetime(dates[1].year, dates[1].month, dates[1].day, 0, 0) + timedelta(
            hours=3 * (1 + (dates[1].hour // 3)))
        one = False
        wind_srcfiles = []
        for wind_file in sorted(os.listdir(dirpath)):
            wind_date = datetime(int(wind_file[:4]), int(wind_file[4:6]), int(wind_file[6:8]))
            if wind_date < start_date_rounded:
                continue
            if wind_date >= stop_date_rounded:
                continue
            if one and single_frame:
                break
            wind_srcfiles.append(wind_file)

        for wind_file in tqdm(wind_srcfiles):
            grb = pygrib.open(os.path.join(dirpath, wind_file))
            grb_u = grb.select(name='U component of wind')
            grb_v = grb.select(name='V component of wind')
            if i_member is None:
                r = range(len(grb_u))
            else:
                fr_time = len(grb_u) // 10
                r = [i_member + i * 10 for i in range(fr_time)]
            for i in r:
                date = f'{grb_u[i]["date"]:0>8}'
                hours = f'{grb_u[i]["time"]:0>4}'
                t = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]),
                             int(hours[:2]), int(hours[2:4]))
                if t < startd_rounded:
                    continue
                if t > stopd_rounded:
                    continue
                if one and single_frame:
                    break
                one = True
                uv = np.zeros(shape + (2,))
                if lon_b[1] > 360:
                    U1 = grb_u[i].data(lat1=bl_d[1], lat2=tr_d[1], lon1=lon_b[0], lon2=360)[0].transpose()
                    V1 = grb_v[i].data(lat1=bl_d[1], lat2=tr_d[1], lon1=lon_b[0], lon2=360)[0].transpose()
                    U2 = grb_u[i].data(lat1=bl_d[1], lat2=tr_d[1], lon1=0, lon2=lon_b[1] - 360)[0].transpose()
                    V2 = grb_v[i].data(lat1=bl_d[1], lat2=tr_d[1], lon1=0, lon2=lon_b[1] - 360)[0].transpose()
                    U = np.concatenate((U1, U2))
                    V = np.concatenate((V1, V2))
                else:
                    U = grb_u[i].data(lat1=bl_d[1], lat2=tr_d[1], lon1=lon_b[0], lon2=lon_b[1])[0].transpose()
                    V = grb_v[i].data(lat1=bl_d[1], lat2=tr_d[1], lon1=lon_b[0], lon2=lon_b[1])[0].transpose()
                uv[:, ::-1, 0] = U
                uv[:, ::-1, 1] = V
                uv_frames.append(uv)
                ts.append(t.timestamp())

        values = np.array(uv_frames)

        return cls(values, np.column_stack((np.array((ts[0], ts[-1])), grid_bounds.transpose())).transpose(),
                   Utils.COORD_GCS, grad_values=None)

    def value(self, t, x):
        """
        Return an interpolated wind value based on linear interpolation
        :param t: Time stamp
        :param x: Position
        :return: Interpolated wind vector
        """
        return Utils.interpolate(self.values, self.bounds.transpose()[0], self.spacings, np.array((t,) + tuple(x)))

    def d_value(self, t, x):
        """
        Return an interpolated wind gradient value based on linear interpolation
        :param t: Time stamp
        :param x: Position
        :return: Interpolated wind jacobian at requested time and position
        """
        return Utils.interpolate(self.grad_values, self.bounds.transpose()[0], self.spacings, np.array((t,) + tuple(x)),
                                 ndim_values_data=2)

    def compute_derivatives(self):
        """
        Computes the derivatives of the windfield with a central difference scheme
        on the wind native grid
        """
        self.grad_values = np.zeros(self.values.shape + (2,))

        nt, nx, ny, _ = self.values.shape
        # Use order 2 precision derivative
        factor = 1 / Utils.EARTH_RADIUS if self.coords == 'gcs' else 1.
        self.grad_values[:, 1:-1, 1:-1, ...] = factor * (
            np.stack(((self.values[:, 2:, 1:-1, 0] - self.values[:, :-2, 1:-1, 0]) / (2 * self.spacings[1]),
                      (self.values[:, 1:-1, 2:, 0] - self.values[:, 1:-1, :-2, 0]) / (2 * self.spacings[2]),
                      (self.values[:, 2:, 1:-1, 1] - self.values[:, :-2, 1:-1, 1]) / (2 * self.spacings[1]),
                      (self.values[:, 1:-1, 2:, 1] - self.values[:, 1:-1, :-2, 1]) / (2 * self.spacings[2])),
                     axis=-1
                     ).reshape((nt, nx - 2, ny - 2, 2, 2))
        )
        # Padding to full grid
        # Borders
        self.grad_values[:, 0, 1:-1, ...] = self.grad_values[:, 1, 1:-1, ...]
        self.grad_values[:, -1, 1:-1, ...] = self.grad_values[:, -2, 1:-1, ...]
        self.grad_values[:, 1:-1, 0, ...] = self.grad_values[:, 1:-1, 1, ...]
        self.grad_values[:, 1:-1, -1, ...] = self.grad_values[:, 1:-1, -2, ...]
        # Corners
        self.grad_values[:, 0, 0, ...] = self.grad_values[:, 1, 1, ...]
        self.grad_values[:, 0, -1, ...] = self.grad_values[:, 1, -2, ...]
        self.grad_values[:, -1, 0, ...] = self.grad_values[:, -2, 1, ...]
        self.grad_values[:, -1, -1, ...] = self.grad_values[:, -2, -2, ...]

    # def flatten(self, proj='plate-carree', **kwargs):
    #     """
    #     Flatten GCS-coordinates wind to euclidean space using given projection
    #     :return The flattened wind
    #     """
    #     if self.coords != Utils.COORD_GCS:
    #         print('Coordinates must be GCS to flatten', file=sys.stderr)
    #         exit(1)
    #     oldgrid = np.zeros(self.grid.shape)
    #     oldgrid[:] = self.grid
    #     oldbounds = (self.x_min, self.y_min, self.x_max, self.y_max)
    #     if proj == 'plate-carree':
    #         center = 0.5 * (self.grid[0, 0] + self.grid[-1, -1])
    #         newgrid = np.zeros(self.grid.shape)
    #         newgrid[:] = Utils.EARTH_RADIUS * (self.grid - center)
    #         self.grid[:] = newgrid
    #         self.x_min = self.grid[0, 0, 0]
    #         self.y_min = self.grid[0, 0, 1]
    #         self.x_max = self.grid[-1, -1, 0]
    #         self.y_max = self.grid[-1, -1, 1]
    #         f_wind = DiscreteWind()
    #         nx, ny, _ = self.grid.shape
    #         bl = self.grid[0, 0]
    #         tr = self.grid[-1, -1]
    #         f_wind.load_from_wind(self, nx, ny, bl, tr, coords=Utils.COORD_CARTESIAN)
    #     elif proj == 'ortho':
    #         for p in ['lon_0', 'lat_0']:
    #             if p not in kwargs.keys():
    #                 print(f'Missing parameter {p} for {proj} flatten type', file=sys.stderr)
    #                 exit(1)
    #         # Lon and lats expected in radians
    #         self.lon_0 = kwargs['lon_0']
    #         self.lat_0 = kwargs['lat_0']
    #         # width = kwargs['width']
    #         # height = kwargs['height']
    #
    #         nx, ny, _ = self.grid.shape
    #
    #         # Grid
    #         proj = Proj(proj='ortho', lon_0=Utils.RAD_TO_DEG * self.lon_0, lat_0=Utils.RAD_TO_DEG * self.lat_0)
    #         oldgrid = np.zeros(self.grid.shape)
    #         oldgrid[:] = self.grid
    #         newgrid = np.zeros((2, nx, ny))
    #         newgrid[:] = proj(Utils.RAD_TO_DEG * self.grid[:, :, 0], Utils.RAD_TO_DEG * self.grid[:, :, 1])
    #         self.grid[:] = newgrid.transpose((1, 2, 0))
    #         oldbounds = (self.x_min, self.y_min, self.x_max, self.y_max)
    #         self.x_min = self.grid[:, :, 0].min()
    #         self.y_min = self.grid[:, :, 1].min()
    #         self.x_max = self.grid[:, :, 0].max()
    #         self.y_max = self.grid[:, :, 1].max()
    #
    #         print(self.x_min, self.x_max, self.y_min, self.y_max)
    #
    #         # Wind
    #         newuv = np.zeros(self.uv.shape)
    #         for kt in range(self.uv.shape[0]):
    #             newuv[kt, :] = self._rotate_wind(self.uv[kt, :, :, 0],
    #                                              self.uv[kt, :, :, 1],
    #                                              Utils.RAD_TO_DEG * oldgrid[:, :, 0],
    #                                              Utils.RAD_TO_DEG * oldgrid[:, :, 1])
    #         self.uv[:] = newuv
    #     elif proj == 'omerc':
    #         for p in ['lon_1', 'lat_1', 'lon_2', 'lat_2']:
    #             if p not in kwargs.keys():
    #                 print(f'Missing parameter {p} for {proj} flatten type', file=sys.stderr)
    #                 exit(1)
    #         # Lon and lats expected in radians
    #         self.lon_1 = kwargs['lon_1']
    #         self.lat_1 = kwargs['lat_1']
    #         self.lon_2 = kwargs['lon_2']
    #         self.lat_2 = kwargs['lat_2']
    #         # width = kwargs['width']
    #         # height = kwargs['height']
    #
    #         nx, ny, _ = self.grid.shape
    #
    #         # Grid
    #         proj = Proj(proj='omerc',
    #                     lon_1=Utils.RAD_TO_DEG * self.lon_1,
    #                     lat_1=Utils.RAD_TO_DEG * self.lat_1,
    #                     lon_2=Utils.RAD_TO_DEG * self.lon_2,
    #                     lat_2=Utils.RAD_TO_DEG * self.lat_2,
    #                     )
    #         oldgrid = np.zeros(self.grid.shape)
    #         oldgrid[:] = self.grid
    #         newgrid = np.zeros((2, nx, ny))
    #         newgrid[:] = proj(Utils.RAD_TO_DEG * self.grid[:, :, 0], Utils.RAD_TO_DEG * self.grid[:, :, 1])
    #         self.grid[:] = newgrid.transpose((1, 2, 0))
    #         oldbounds = (self.x_min, self.y_min, self.x_max, self.y_max)
    #         self.x_min = self.grid[:, :, 0].min()
    #         self.y_min = self.grid[:, :, 1].min()
    #         self.x_max = self.grid[:, :, 0].max()
    #         self.y_max = self.grid[:, :, 1].max()
    #
    #         print(self.x_min, self.x_max, self.y_min, self.y_max)
    #
    #         # Wind
    #         newuv = np.zeros(self.uv.shape)
    #         for kt in range(self.uv.shape[0]):
    #             newuv[kt, :] = self._rotate_wind(self.uv[kt, :, :, 0],
    #                                              self.uv[kt, :, :, 1],
    #                                              Utils.RAD_TO_DEG * oldgrid[:, :, 0],
    #                                              Utils.RAD_TO_DEG * oldgrid[:, :, 1])
    #         self.uv[:] = newuv
    #     else:
    #         print(f"Unknown projection type {proj}", file=sys.stderr)
    #         exit(1)
    #
    #     # self.grid[:] = oldgrid
    #     # self.x_min, self.y_min, self.x_max, self.y_max = oldbounds
    #
    #     self.unstructured = True
    #     self.coords = Utils.COORD_CARTESIAN
    #     self.units_grid = 'meters'
    #     self.is_analytical = False
    #
    # def _rotate_wind(self, u, v, x, y):
    #     if self.bm is None:
    #         if self.lon_0 is not None:
    #             self.bm = Basemap(projection='ortho',
    #                               lon_0=Utils.RAD_TO_DEG * self.lon_0,
    #                               lat_0=Utils.RAD_TO_DEG * self.lat_0)
    #         else:
    #             self.bm = Basemap(projection='omerc',
    #                               lon_1=Utils.RAD_TO_DEG * self.lon_1,
    #                               lat_1=Utils.RAD_TO_DEG * self.lat_1,
    #                               lon_2=Utils.RAD_TO_DEG * self.lon_2,
    #                               lat_2=Utils.RAD_TO_DEG * self.lat_2,
    #                               lon_0=Utils.RAD_TO_DEG * 0.5 * (self.lon_1 + self.lon_2),
    #                               lat_0=Utils.RAD_TO_DEG * 0.5 * (self.lat_1 + self.lat_2),
    #                               height=Utils.EARTH_RADIUS,
    #                               width=Utils.EARTH_RADIUS,
    #                               )
    #     return np.array(self.bm.rotate_vector(u, v, x, y)).transpose((1, 2, 0))

    def dualize(self):
        # Override method so that the dual of a DiscreteWind stays a DiscreteWind and
        # is not casted to Wind
        wind = DiscreteWind.from_wind(-1. * self, self.bounds, self.values.shape[1], self.values.shape[2])
        if self.t_end is not None:
            wind.t_start = self.t_end
            wind.t_end = self.t_start
        else:
            wind.t_start = self.t_start
        return wind


class TwoSectorsWind(Wind):

    def __init__(self,
                 v_w1: float,
                 v_w2: float,
                 x_switch: float):
        """
        Wind configuration where wind is constant over two half-planes separated by x = x_switch. x-wind is null.
        :param v_w1: y-wind value for x < x_switch
        :param v_w2: y-wind value for x >= x_switch
        :param x_switch: x-coordinate for sectors separation
        """
        super().__init__()
        self.v_w1 = v_w1
        self.v_w2 = v_w2
        self.x_switch = x_switch

    def value(self, t, x):
        return np.array([0, self.v_w1 * np.heaviside(self.x_switch - x[0], 0.)
                         + self.v_w2 * np.heaviside(x[0] - self.x_switch, 0.)])

    def d_value(self, t, x):
        return np.array([[0, 0],
                         [0, 0]])


class TSEqualWind(TwoSectorsWind):

    def __init__(self, v_w1, v_w2, x_f):
        """
        TwoSectorsWind but the sector separation is midway to the target
        :param x_f: Target x-coordinate.
        """
        super().__init__(v_w1, v_w2, x_f / 2)


class UniformWind(Wind):

    def __init__(self, wind_vector: ndarray):
        """
        :param wind_vector: Direction and strength of wind
        """
        super().__init__()
        self.wind_vector = np.array(wind_vector)

    def _value(self, t, x):
        return self.wind_vector

    def _d_value(self, t, x):
        return np.array([[0., 0.],
                         [0., 0.]])


class VortexWind(Wind):

    def __init__(self,
                 x_omega: float,
                 y_omega: float,
                 gamma: float):
        """
        A vortex from potential flow theory.
        :param x_omega: x_coordinate of vortex center
        :param y_omega: y_coordinate of vortex center
        :param gamma: Circulation of the vortex. Positive is ccw vortex.
        """
        super().__init__()
        self.x_omega = x_omega
        self.y_omega = y_omega
        self.omega = np.array([x_omega, y_omega])
        self.gamma = gamma

    def value(self, t, x):
        r = np.linalg.norm(x - self.omega)
        e_theta = np.array([-(x - self.omega)[1] / r,
                            (x - self.omega)[0] / r])
        return self.gamma / (2 * np.pi * r) * e_theta

    def d_value(self, t, x):
        r = np.linalg.norm(x - self.omega)
        x_omega = self.x_omega
        y_omega = self.y_omega
        return self.gamma / (2 * np.pi * r ** 4) * np.array(
            [[2 * (x[0] - x_omega) * (x[1] - y_omega), (x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2],
             [(x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2, -2 * (x[0] - x_omega) * (x[1] - y_omega)]])


class RankineVortexWind(Wind):

    def __init__(self, center, gamma, radius, t_start=0., t_end=None):
        """
        A vortex from potential theory
        :param center: Coordinates of the vortex center. Shape (2,) if steady else shape (nt, 2)
        :param gamma: Circulation of the vortex. Positive is ccw vortex. Scalar for steady vortex
        else shape (nt,)
        :param radius: Radius of the vortex core. Scalar for steady vortex else shape (nt,)
        """
        nt_int = None
        if len(center.shape) > 1:
            nt_int = center.shape[0]
        if nt_int is not None and t_end is None:
            print('Missing t_end', file=sys.stderr)
            exit(1)
        super().__init__(t_start=t_start, t_end=t_end, nt_int=nt_int)

        self.omega = np.array(center)
        self.gamma = np.array(gamma)
        self.radius = np.array(radius)
        if nt_int is not None and not (self.omega.shape[0] == self.gamma.shape[0] == self.radius.shape[0]):
            print('Incoherent sizes', file=sys.stderr)
            exit(1)
        self.zero_ceil = 1e-3

    def max_speed(self, t=0.):
        _, gamma, radius = self.params(t)
        return gamma / (radius * 2 * np.pi)

    def params(self, t):
        if self.t_end is None:
            return self.omega, self.gamma, self.radius
        i, alpha = self._index(t)
        omega = (1 - alpha) * self.omega[i] + alpha * self.omega[i + 1]
        gamma = (1 - alpha) * self.gamma[i] + alpha * self.gamma[i + 1]
        radius = (1 - alpha) * self.radius[i] + alpha * self.radius[i + 1]
        return omega, gamma, radius

    def value(self, t, x):
        omega, gamma, radius = self.params(t)
        r = np.linalg.norm(x - omega)
        if r < self.zero_ceil * radius:
            return np.zeros(2)
        e_theta = np.array([-(x - omega)[1] / r,
                            (x - omega)[0] / r])
        f = gamma / (2 * np.pi)
        if r <= radius:
            return f * r / (radius ** 2) * e_theta
        else:
            return f * 1 / r * e_theta

    def d_value(self, t, x):
        omega, gamma, radius = self.params(t)
        r = np.linalg.norm(x - omega)
        if r < self.zero_ceil * radius:
            return np.zeros((2, 2))
        x_omega = omega[0]
        y_omega = omega[1]
        # e_r = np.array([(x - omega)[0] / r,
        #                 (x - omega)[1] / r])
        e_theta = np.array([-(x - omega)[1] / r,
                            (x - omega)[0] / r])
        f = gamma / (2 * np.pi)
        if r <= radius:
            a = (x - omega)[0]
            b = (x - omega)[1]
            d_e_theta__d_x = np.array([a * b / r ** 3, b ** 2 / r ** 3])
            d_e_theta__d_y = np.array([- a ** 2 / r ** 3, - a * b / r ** 3])
            return f / radius ** 2 * np.stack((a / r * e_theta + r * d_e_theta__d_x,
                                               b / r * e_theta + r * d_e_theta__d_y), axis=1)
        else:
            return f / r ** 4 * \
                   np.array([[2 * (x[0] - x_omega) * (x[1] - y_omega), (x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2],
                             [(x[1] - y_omega) ** 2 - (x[0] - x_omega) ** 2, -2 * (x[0] - x_omega) * (x[1] - y_omega)]])


# class VortexBarrierWind(Wind):
#     def __init__(self,
#                  x_omega: float,
#                  y_omega: float,
#                  gamma: float,
#                  radius: float):
#         """
#         A vortex from potential theory, but add a wind barrier at a given radius
#
#         :param x_omega: x_coordinate of vortex center in m
#         :param y_omega: y_coordinate of vortex center in m
#         :param gamma: Circulation of the vortex in m^2/s. Positive is ccw vortex.
#         :param radius: Radius of the barrier in m
#         """
#         super().__init__(value_func=self.value, d_value_func=self.d_value)
#         self.x_omega = x_omega
#         self.y_omega = y_omega
#         self.omega = np.array([x_omega, y_omega])
#         self.gamma = gamma
#         self.radius = radius
#         self.zero_ceil = 1e-3
#         self.descr = f'Vortex barrier'  # ({self.gamma:.2f} m^2/s, at ({self.x_omega:.2f}, {self.y_omega:.2f}))'
#         self.is_analytical = True

class LinearWind(Wind):
    """
    Constant gradient wind
    """

    def __init__(self, gradient: ndarray, origin: ndarray, value_origin: ndarray):
        """
        :param gradient: The wind gradient
        :param origin: The origin point for the origin value
        :param value_origin: The origin value
        """
        super().__init__()

        self.gradient = np.zeros(gradient.shape)
        self.origin = np.zeros(origin.shape)
        self.value_origin = np.zeros(value_origin.shape)

        self.gradient[:] = gradient
        self.origin[:] = origin
        self.value_origin[:] = value_origin

    def value(self, t, x):
        return self.gradient.dot(x - self.origin) + self.value_origin

    def d_value(self, t, x):
        return self.gradient


# TODO: Unite with steady linear
class LinearWindT(Wind):
    """
    Gradient does not vary with space but does vary with time
    """

    def __init__(self, gradient, origin, value_origin, t_end):
        """
        :param gradient: The wind gradient, shape (nt, 2, 2)
        :param origin: The origin point for the origin value, shape (2,)
        :param value_origin: The origin value, shape (2,)
        """
        super().__init__(t_end=t_end)

        self.gradient = np.zeros(gradient.shape)
        self.origin = np.zeros(origin.shape)
        self.value_origin = np.zeros(value_origin.shape)

        self.gradient[:] = gradient
        self.origin[:] = origin
        self.value_origin[:] = value_origin

    def value(self, t, x):
        i, alpha = self._index(t)
        return ((1 - alpha) * self.gradient[i] + alpha * self.gradient[i + 1]) @ (x - self.origin) + self.value_origin

    def d_value(self, t, x):
        i, alpha = self._index(t)
        return (1 - alpha) * self.gradient[i] + alpha * self.gradient[i + 1]

    def dualize(self):
        return -1. * LinearWindT(self.gradient[::-1, :, :], self.origin, self.value_origin, self.t_end)


class PointSymWind(Wind):
    """
    From Techy 2011 (DOI 10.1007/s11370-011-0092-9)
    """

    def __init__(self, x_center: float, y_center: float, gamma: float, omega: float):
        """
        :param x_center: Center x-coordinate
        :param y_center: Center y-coordinate
        :param gamma: Divergence
        :param omega: Curl
        """
        super().__init__()

        self.center = np.array((x_center, y_center))

        self.gamma = gamma
        self.omega = omega
        self.mat = np.array([[gamma, -omega], [omega, gamma]])

    def value(self, t, x):
        return self.mat @ (x - self.center)

    def d_value(self, t, x):
        return self.mat


class DoubleGyreWind(Wind):
    """
    From Li 2020 (DOI 10.1109/JOE.2019.2926822)
    """

    def __init__(self, x_center: float, y_center: float, x_wl: float, y_wl: float, ampl: float):
        """
        :param x_center: Bottom left x-coordinate
        :param y_center: Bottom left y-coordinate
        :param x_wl: x-axis wavelength
        :param y_wl: y-axis wavelength
        :param ampl: Amplitude in meters per second
        """
        super().__init__()

        self.center = np.array((x_center, y_center))

        # Phase gradient
        self.kx = 2 * pi / x_wl
        self.ky = 2 * pi / y_wl
        self.ampl = ampl

    def value(self, t, x):
        xx = np.diag((self.kx, self.ky)) @ (x - self.center)
        return self.ampl * np.array((-sin(xx[0]) * cos(xx[1]), cos(xx[0]) * sin(xx[1])))

    def d_value(self, t, x):
        xx = np.diag((self.kx, self.ky)) @ (x - self.center)
        return self.ampl * np.array([[-self.kx * cos(xx[0]) * cos(xx[1]), self.ky * sin(xx[0]) * sin(xx[1])],
                                     [-self.kx * sin(xx[0]) * sin(xx[1]), self.ky * cos(xx[0]) * cos(xx[1])]])


class DoubleGyreDampedWind(Wind):

    def __init__(self,
                 x_center: float,
                 y_center: float,
                 x_wl: float,
                 y_wl: float,
                 ampl: float,
                 x_center_damp: float,
                 y_center_damp: float,
                 lambda_x_damp: float,
                 lambda_y_damp: float):
        super().__init__()
        self.double_gyre = DoubleGyreWind(x_center, y_center, x_wl, y_wl, ampl)
        self.center_damp = np.array((x_center_damp, y_center_damp))
        self.lambda_x_damp = lambda_x_damp
        self.lambda_y_damp = lambda_y_damp

    def value(self, t, x):
        xx = np.diag((1 / self.lambda_x_damp, 1 / self.lambda_y_damp)) @ (x - self.center_damp)
        damp = 1 / (1 + xx[0] ** 2 + xx[1] ** 2)
        return self.double_gyre.value(t, x) * damp


class RadialGaussWind(Wind):
    def __init__(self, x_center: float, y_center: float, radius: float, sdev: float, v_max: float):
        """
        :param x_center: Center x-coordinate
        :param y_center: Center y-coordinate
        :param radius: Radial distance of maximum wind value (absolute value)
        :param sdev: Gaussian standard deviation
        :param v_max: Maximum wind value
        """
        super().__init__()

        self.center = np.array((x_center, y_center))

        self.radius = radius
        self.sdev = sdev
        self.v_max = v_max
        self.zero_ceil = 1e-3
        self.dualizable = False

    def ampl(self, r):
        if r < self.zero_ceil * self.radius:
            return 0.
        return self.v_max * exp(-(log(r / self.radius)) ** 2 / (2 * self.sdev ** 2))

    def value(self, t, x):
        xx = x - self.center
        r = np.linalg.norm(xx)
        if r < self.zero_ceil * self.radius:
            return np.zeros(2)
        e_r = (x - self.center) / r
        v = self.ampl(r)
        return v * e_r

    def d_value(self, t, x):
        xx = x - self.center
        r = np.linalg.norm(xx)
        if r < self.zero_ceil * self.radius:
            return np.zeros((2, 2))
        e_r = (x - self.center) / r
        a = (x - self.center)[0]
        b = (x - self.center)[1]
        dv = -log(r / self.radius) * self.ampl(r) / (r ** 2 * self.sdev ** 2) * np.array([xx[0], xx[1]])
        nabla_e_r = np.array([[b ** 2 / r ** 3, - a * b / r ** 3],
                              [-a * b / r ** 3, a ** 2 / r ** 3]])
        return np.einsum('i,j->ij', e_r, dv) + self.ampl(r) * nabla_e_r


class RadialGaussWindT(Wind):
    """
    Time-varying version of radial Gauss wind
    """

    def __init__(self, center, radius, sdev, v_max, t_end, t_start=0.):
        """
        :param center: ndarray of size nt x 2
        :param radius: ndarray of size nt
        :param sdev: ndarray of size nt
        :param v_max: ndarray of size nt
        :param t_end: end of time window. Wind is supposed to be regularly sampled in the time window
        :param t_start: beginning of time window.
        """
        super().__init__(t_start=t_start, t_end=t_end,
                         nt_int=radius.shape[0])

        self.center = np.zeros(center.shape)
        self.radius = np.zeros(radius.shape)
        self.sdev = np.zeros(sdev.shape)
        self.v_max = np.zeros(v_max.shape)

        self.center[:] = center
        self.radius[:] = radius
        self.sdev[:] = sdev
        self.v_max[:] = v_max

        self.zero_ceil = 1e-3
        self.is_analytical = True
        self.dualizable = False

    def ampl(self, t, r):
        i, alpha = self._index(t)
        radius = (1 - alpha) * self.radius[i] + alpha * self.radius[i + 1]
        v_max = (1 - alpha) * self.v_max[i] + alpha * self.v_max[i + 1]
        sdev = (1 - alpha) * self.sdev[i] + alpha * self.sdev[i + 1]
        if r < self.zero_ceil * radius:
            return 0.
        return v_max * exp(-(log(r / radius)) ** 2 / (2 * sdev ** 2))

    def value(self, t, x):
        i, alpha = self._index(t)
        center = (1 - alpha) * self.center[i] + alpha * self.center[i + 1]
        radius = (1 - alpha) * self.radius[i] + alpha * self.radius[i + 1]
        xx = x - center
        r = np.linalg.norm(xx)
        if r < self.zero_ceil * radius:
            return np.zeros(2)
        e_r = (x - center) / r
        v = self.ampl(t, r)
        return v * e_r

    def d_value(self, t, x):
        i, alpha = self._index(t)
        center = (1 - alpha) * self.center[i] + alpha * self.center[i + 1]
        radius = (1 - alpha) * self.radius[i] + alpha * self.radius[i + 1]
        sdev = (1 - alpha) * self.sdev[i] + alpha * self.sdev[i + 1]
        xx = x - center
        r = np.linalg.norm(xx)
        if r < self.zero_ceil * radius:
            return np.zeros((2, 2))
        e_r = (x - center) / r
        a = (x - center)[0]
        b = (x - center)[1]
        dv = -log(r / radius) * self.ampl(t, r) / (r ** 2 * sdev ** 2) * np.array([xx[0], xx[1]])
        nabla_e_r = np.array([[b ** 2 / r ** 3, - a * b / r ** 3],
                              [-a * b / r ** 3, a ** 2 / r ** 3]])
        return np.einsum('i,j->ij', e_r, dv) + self.ampl(t, r) * nabla_e_r


class BandGaussWind(Wind):
    """
    Linear band of gaussian wind
    """

    def __init__(self, origin, vect, ampl, sdev):
        super().__init__()
        self.origin = np.zeros(2)
        self.origin[:] = origin
        self.vect = np.zeros(2)
        self.vect = vect / np.linalg.norm(vect)
        self.ampl = ampl
        self.sdev = sdev

    def value(self, t, x):
        dist = np.abs(np.cross(self.vect, x - self.origin))
        intensity = self.ampl * np.exp(-0.5 * (dist / self.sdev) ** 2)
        return self.vect * intensity

    def d_value(self, t, x):
        # TODO : write analytical formula
        dx = 1e-6
        return np.column_stack((1 / dx * (self.value(t, x + np.array((dx, 0.))) - self.value(t, x)),
                                1 / dx * (self.value(t, x + np.array((0., dx))) - self.value(t, x))))


class BandWind(Wind):
    """
    Band of wind. NON DIFFERENTIABLE (should be instanciated as discrete wind)
    """

    def __init__(self, origin, vect, w_value, width):
        super().__init__()
        self.origin = np.zeros(2)
        self.origin[:] = origin
        self.vect = np.zeros(2)
        self.vect = vect / np.linalg.norm(vect)
        self.w_value = w_value
        self.width = width

    def value(self, t, x):
        dist = np.abs(np.cross(self.vect, x - self.origin))
        if dist >= self.width / 2.:
            return np.zeros(2)
        else:
            return self.w_value

    def d_value(self, t, x):
        print('Undefined', file=sys.stderr)
        exit(1)


class LCWind(Wind):
    """
    Linear combination of winds.
    """

    def __init__(self, coeffs, winds):
        self.coeffs = np.zeros(coeffs.shape[0])
        self.coeffs[:] = coeffs
        self.winds = winds

        self.lcwind = sum((c * self.winds[i] for i, c in enumerate(self.coeffs)), UniformWind(np.zeros(2)))
        nt_int = None
        t_start = None
        t_end = None
        for wind in self.winds:
            if wind.nt_int is not None and wind.nt_int > 1:
                if nt_int is None:
                    nt_int = wind.nt_int
                    t_start = wind.t_start
                    t_end = wind.t_end
                else:
                    if wind.nt != nt_int or wind.t_start != t_start or wind.t_end != t_end:
                        print('Cannot handle combination of multiple time-varying winds with '
                              'different parameters for the moment', file=sys.stderr)
                        exit(1)
        super().__init__(t_start, t_end, nt_int)

    def value(self, t, x):
        return self.lcwind.value(t, x)

    def d_value(self, t, x):
        return self.lcwind.d_value(t, x)

    def dualize(self):
        new_coeffs = np.zeros(self.coeffs.shape[0])
        for i, c in enumerate(self.coeffs):
            if self.winds[i].dualizable:
                new_coeffs[i] = -1. * c
            else:
                new_coeffs[i] = c
        return LCWind(new_coeffs, self.winds)


class LVWind(Wind):
    """
    Wind varying linearly with time. Wind is spaitially uniform at each timestep.
    """

    def __init__(self, wind_value, gradient, time_scale):
        super().__init__()
        self.wind_value = np.array(wind_value)
        self.gradient = np.array(gradient)
        self.t_end = time_scale

    def value(self, t, x):
        return self.wind_value + t * self.gradient

    def d_value(self, t, x):
        return np.zeros(2)


class TrapWind(Wind):
    def __init__(self, wind_value, center, radius, rel_wid=0.05, t_start=0., t_end=None):
        """
        :param wind_value: (nt,) in meters per second
        :param center: (nt, 2) in meters
        :param radius: (nt,) in meters
        """
        if wind_value.shape[0] != radius.shape[0]:
            raise Exception('Incoherent shapes')
        super().__init__(t_start=t_start, t_end=t_end, nt_int=wind_value.shape[0])
        self.wind_value = np.array(wind_value)
        self.center = np.array(center)
        self.radius = np.array(radius)
        # Relative obstacle variation width
        self.rel_wid = rel_wid

    @staticmethod
    def sigmoid(r, wid):
        return 1 / (1 + np.exp(-4 / wid * r))

    @staticmethod
    def d_sigmoid(r, wid):
        lam = 4 / wid
        s = TrapWind.sigmoid(r, wid)
        return lam * s * (1 - s)

    def value(self, t, x):
        i, alpha = self._index(t)
        wind_value = (1 - alpha) * self.wind_value[i] + (0. if alpha < 1e-3 else alpha * self.wind_value[i + 1])
        center = (1 - alpha) * self.center[i] + (0. if alpha < 1e-3 else alpha * self.center[i + 1])
        radius = (1 - alpha) * self.radius[i] + (0. if alpha < 1e-3 else alpha * self.radius[i + 1])
        if (x - center) @ (x - center) < 1e-8 * radius ** 2:
            return np.zeros(2)
        else:
            r = np.linalg.norm((x - center))
            e_r = (x - center) / r
            return -wind_value * TrapWind.sigmoid(r - radius, radius * self.rel_wid) * e_r

    def d_value(self, t, x):
        i, alpha = self._index(t)
        wind_value = (1 - alpha) * self.wind_value[i] + (0. if alpha < 1e-3 else alpha * self.wind_value[i + 1])
        center = (1 - alpha) * self.center[i] + (0. if alpha < 1e-3 else alpha * self.center[i + 1])
        radius = (1 - alpha) * self.radius[i] + (0. if alpha < 1e-3 else alpha * self.radius[i + 1])
        if (x - center) @ (x - center) < 1e-8 * radius ** 2:
            return np.zeros((2, 2))
        else:
            r = np.linalg.norm((x - center))
            e_r = (x - center) / r
            P = np.array(((e_r[0], e_r[1]), (-e_r[1] / r, e_r[0] / r)))
            return -wind_value * (P.transpose() @ np.diag((TrapWind.d_sigmoid(r - radius, radius * self.rel_wid),
                                                           TrapWind.sigmoid(r - radius, radius * self.rel_wid))) @ P)


class ChertovskihWind(Wind):

    def __init__(self):
        super().__init__()

    def value(self, t, x):
        return np.array((x[0] * (x[1] + 3) / 4, -x[0] * x[0]))

    def d_value(self, t, x):
        return np.array((((x[1] + 3) / 4, x[0] / 4),
                         (-2 * x[0], 0)))
