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
flowfield.py
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


class FlowField(ABC):

    def __init__(self, _lch=None, _rch=None, _op=None, t_start=0., t_end=None, nt_int=None,
                 coords=Utils.COORD_CARTESIAN):
        self._lch = _lch
        self._rch = _rch
        self._op = _op

        # When flow field is time-varying, bounds for the time window
        # A none upper bound means no time variation
        self.t_start = t_start
        self.t_end = t_end

        # Intrisic number of time frames when analytically defined
        # flow fields depend on time-varying parameters
        self.nt_int = nt_int

        # True if dualization operation flips flow field (mulitplication by -1)
        # False if dualization operation leaves flow field unchanged
        self.dualizable = True

        self.coords = coords

    def value(self, t, x):
        if self._lch is None:
            return self._value(t, x)
        if self._op == '+':
            return self._lch.value(t, x) + self._rch.value(t, x)
        if self._op == '-':
            return self._lch.value(t, x) - self._rch.value(t, x)
        if self._op == '*':
            if isinstance(self._lch, float):
                return self._lch * self._rch.value(t, x)
            if isinstance(self._rch, float):
                return self._lch.value(t, x) * self._rch
            raise Exception('Only scaling by float implemented for flow fields')

    def d_value(self, t, x):
        if self._lch is None or self._rch is None:
            return self._d_value(t, x)
        if self._op == '+':
            return self._lch.d_value(t, x) + self._rch.d_value(t, x)
        if self._op == '-':
            return self._lch.d_value(t, x) - self._rch.d_value(t, x)
        if self._op == '*':
            if isinstance(self._lch, float):
                return self._lch * self._rch.d_value(t, x)
            if isinstance(self._rch, float):
                return self._lch.d_value(t, x) * self._rch
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
        Add flow fields
        :param other: Another flow field
        :return: The sum of the two flow fields
        """
        if not isinstance(other, FlowField):
            raise TypeError(f"Unsupported type for addition : {type(other)}")
        return FlowField(_lch=self, _rch=other, _op='+')

    def __sub__(self, other):
        """
        Substracts flow fields
        :param other: Another flow fields
        :return: The substraction of the two flow fields
        """
        if not isinstance(other, FlowField):
            raise TypeError(f"Unsupported type for substraction : {type(other)}")
        return FlowField(_lch=self, _rch=other, _op='-')

    def __mul__(self, other):
        """
        Handles the scaling of a flow field by a real number
        :param other: A real number (float)
        :return: The scaled flow field
        """
        if isinstance(other, int):
            other = float(other)
        if not isinstance(other, float):
            raise TypeError(f"Unsupported type for multiplication : {type(other)}")
        return FlowField(_lch=self, _rch=other, _op='*', t_start=self.t_start, t_end=self.t_end)

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
        # Bounds may not be in the right order if flow field is dualized
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


class DiscreteFF(FlowField):
    """
    Handles flow field loading from H5 format and derivative computation
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

        if values.ndim == 3:
            self.value = self._value_steady
            self.d_value = self._d_value_steady
        else:
            self.value = self._value_unsteady
            self.d_value = self._d_value_unsteady
            self.t_start = bounds[0, 0]
            self.t_end = bounds[0, 1]

    @classmethod
    def from_npz(cls, filepath):
        ff = np.load(filepath)
        return cls(ff['values'], ff['bounds'], str(ff['coords']))

    @classmethod
    def from_h5(cls, filepath, **kwargs):
        """
        Loads flow field data from H5 flow field data
        :param filepath: The H5 file contaning flow field data
        """

        with h5py.File(filepath, 'r') as ff_data:
            coords = ff_data.attrs['coords']

            # Checking consistency before loading
            Utils.ensure_coords(coords)

            values = np.array(ff_data['data']).squeeze()

            # Time bounds
            t_start = ff_data['ts'][0]
            t_end = None if ff_data['ts'].shape[0] == 1 else ff_data['ts'][-1]

            # Detecting millisecond-formated timestamps
            if np.any(np.array(ff_data['ts']) > 1e11):
                t_start /= 1000.
                if t_end is not None:
                    t_end /= 1000.
            f = Utils.DEG_TO_RAD if ff_data.attrs['units_grid'] == Utils.U_DEG else 1.

            bounds = np.stack((() if t_end is None else (np.array((t_start, t_end)),)) +
                              (f * ff_data['grid'][0, 0],
                               f * ff_data['grid'][-1, -1]), axis=0)

        return cls(values, bounds, coords, **kwargs)

    @classmethod
    def from_ff(cls, ff: FlowField, grid_bounds: Union[tuple[ndarray, ndarray], ndarray],
                nx=100, ny=100, nt=50, coords: Optional[str] = None, **kwargs):
        """
        Create discrete flow field by sampling another flow field
        :param ff: Flow field
        :param grid_bounds: if tuple, must be (bl, tr) with bl the bottom left corner vector and tr
        the top right corner vector. If array, must be a (2, 2) array with array[0] being min an max values
        for first coordinate and array[1] the min and max values for second coordinate (this is the transpose of the
        tuple version)
        :param nx: First dimension discretization number
        :param ny: Second dimension discretization number
        :param nt: Time dimension discretization number
        :param kwargs: Additional kwargs
        :return: A DiscreteFF object
        """
        t_start = ff.t_start
        t_end = ff.t_end
        if isinstance(grid_bounds, tuple):
            if len(grid_bounds) != 2:
                raise ValueError('"grid_bounds" provided as a tuple must have two elements')
            grid_bounds = np.array(grid_bounds).transpose()

        if isinstance(grid_bounds, ndarray) and grid_bounds.shape != (2, 2):
            raise ValueError('"grid_bounds" provided as an array must have shape (2, 2)')

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
                        values[i, j, :] = ff.value(t, state)
                    else:
                        values[k, i, j, :] = ff.value(t, state)
                    if not kwargs.get('force_no_diff'):
                        if t_end is None:
                            grad_values[i, j, ...] = ff.d_value(t, state)
                        else:
                            grad_values[k, i, j, ...] = ff.d_value(t, state)

        coords = coords if coords is not None else \
            Utils.COORD_GCS if hasattr(ff, 'coords') and ff.coords == Utils.COORD_GCS \
                else Utils.COORD_CARTESIAN
        return cls(values, bounds, coords, grad_values=grad_values, **kwargs)

    @classmethod
    def from_cds(cls, grid_bounds: ndarray, t_start: Union[float | datetime], t_end: Union[float | datetime],
                 i_member: Optional[int] = None, resolution='0.5', pressure_level='1000',
                 data_path: Optional[str] = None):
        """
        :param grid_bounds: A (2,2) array where the zeroth element corresponds to the bounds of the zeroth axis
        and the first element to the bounds of the first axis
        :param t_start: The required start time for flow field
        :param t_end: The required end time for flow field
        :param i_member: The reanalysis ensemble member number
        :param resolution: The weather model grid resolution in degrees, e.g. '0.5'
        :param pressure_level: The pressure level in hPa, e.g. '1000', '500', '200'
        :param data_path: Force path to data to this value
        :return: A DiscreteFF corresponding to the query
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
        Return an interpolated flow field value based on linear interpolation
        :param t: Time stamp
        :param x: Position
        :return: Interpolated flow field vector
        """
        pass

    def _value_steady(self, _, x: ndarray):
        return Utils.interpolate(self.values, self.bounds.transpose()[0], self.spacings, x)

    def _value_unsteady(self, t, x):
        return Utils.interpolate(self.values, self.bounds.transpose()[0], self.spacings, np.array((t,) + tuple(x)))

    def d_value(self, t, x):
        """
        Return an interpolated flow field gradient value based on linear interpolation
        :param t: Time stamp
        :param x: Position
        :return: Interpolated flow field jacobian at requested time and position
        """
        pass

    def _d_value_steady(self, _, x):
        return Utils.interpolate(self.grad_values, self.bounds.transpose()[0], self.spacings, x,
                                 ndim_values_data=2)

    def _d_value_unsteady(self, t, x):
        return Utils.interpolate(self.grad_values, self.bounds.transpose()[0], self.spacings, np.array((t,) + tuple(x)),
                                 ndim_values_data=2)

    def compute_derivatives(self):
        """
        Computes the derivatives of the flow field with a central difference scheme
        on the flow field native grid
        """
        grad_shape = self.values.shape + (2,)
        self.grad_values = np.zeros(grad_shape)
        inside_shape = np.array(grad_shape, dtype=int)
        inside_shape[-4] -= 2  # x-axis
        inside_shape[-3] -= 2  # y-axis
        inside_shape = tuple(inside_shape)

        # Use order 2 precision derivative
        self.grad_values[..., 1:-1, 1:-1, :, :] = \
            np.stack(((self.values[..., 2:, 1:-1, 0] - self.values[..., :-2, 1:-1, 0]) / (2 * self.spacings[-2]),
                      (self.values[..., 1:-1, 2:, 0] - self.values[..., 1:-1, :-2, 0]) / (2 * self.spacings[-1]),
                      (self.values[..., 2:, 1:-1, 1] - self.values[..., :-2, 1:-1, 1]) / (2 * self.spacings[-2]),
                      (self.values[..., 1:-1, 2:, 1] - self.values[..., 1:-1, :-2, 1]) / (2 * self.spacings[-1])),
                     axis=-1
                     ).reshape(inside_shape)
        # Padding to full grid
        # Borders
        self.grad_values[..., 0, 1:-1, :, :] = self.grad_values[..., 1, 1:-1, :, :]
        self.grad_values[..., -1, 1:-1, :, :] = self.grad_values[..., -2, 1:-1, :, :]
        self.grad_values[..., 1:-1, 0, :, :] = self.grad_values[..., 1:-1, 1, :, :]
        self.grad_values[..., 1:-1, -1, :, :] = self.grad_values[..., 1:-1, -2, :, :]
        # Corners
        self.grad_values[..., 0, 0, :, :] = self.grad_values[..., 1, 1, :, :]
        self.grad_values[..., 0, -1, :, :] = self.grad_values[..., 1, -2, :, :]
        self.grad_values[..., -1, 0, :, :] = self.grad_values[..., -2, 1, :, :]
        self.grad_values[..., -1, -1, :, :] = self.grad_values[..., -2, -2, :, :]

    def dualize(self):
        # Override method so that the dual of a DiscreteFF stays a DiscreteFF and
        # is not casted to FlowField
        ff = DiscreteFF.from_ff(-1. * self, self.bounds, self.values.shape[1], self.values.shape[2])
        if self.t_end is not None:
            ff.t_start = self.t_end
            ff.t_end = self.t_start
        else:
            ff.t_start = self.t_start
        return ff


class WrapperFF(FlowField):
    """
    Wrap a flow field to work with appropriate units
    """

    def __init__(self, ff: FlowField, scale_length: float, bl: ndarray,
                 scale_time: float, time_origin: float):
        self.ff: FlowField = ff
        self.scale_length = scale_length
        self.bl: ndarray = bl.copy()
        self.scale_time = scale_time
        self.time_origin = time_origin
        # In GCS convention, flow field magnitude is in meters per seconds
        # and should be cast to radians per seconds before regular scaling
        self.scale_speed = scale_length / scale_time * (Utils.EARTH_RADIUS if ff.coords == Utils.COORD_GCS else 1.)
        # Multiplication will be quicker than division
        self._scaler_speed = 1 / self.scale_speed
        super().__init__(coords=ff.coords, t_start=(ff.t_start - self.time_origin)/self.scale_time,
                         t_end=None if ff.t_end is None else (ff.t_end - self.time_origin)/self.scale_time)

    def value(self, t, x):
        return self._scaler_speed * self.ff.value(self.time_origin + t * self.scale_time,
                                                  self.bl + x * self.scale_length)

    def d_value(self, t, x):
        return self._scaler_speed * self.ff.d_value(self.time_origin + t * self.scale_time,
                                                    self.bl + x * self.scale_length)

    def __getattr__(self, item):
        return self.ff.__getattribute__(item)


class TwoSectorsFF(FlowField):

    def __init__(self,
                 v_w1: float,
                 v_w2: float,
                 x_switch: float):
        """
        Flow field configuration where flow field is constant over two half-planes separated by x = x_switch.
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


class TSEqualFF(TwoSectorsFF):

    def __init__(self, v_w1, v_w2, x_f):
        """
        TwoSectorsFF but the sector separation is midway to the target
        :param x_f: Target x-coordinate.
        """
        super().__init__(v_w1, v_w2, x_f / 2)


class UniformFF(FlowField):

    def __init__(self, ff_val: ndarray):
        """
        :param ff_val: Direction and strength of flow field
        """
        super().__init__()
        self.ff_val = ff_val.copy()

    def _value(self, t, x):
        return self.ff_val

    def _d_value(self, t, x):
        return np.array([[0., 0.],
                         [0., 0.]])


class ZeroFF(UniformFF):
    def __init__(self):
        super(ZeroFF, self).__init__(np.zeros(2))


class VortexFF(FlowField):

    def __init__(self,
                 center: ndarray,
                 circulation: float):
        """
        A vortex from potential flow theory.
        :param center: 2D vector of center coordinates
        :param circulation: Circulation of the vortex. Positive is ccw vortex.
        """
        super().__init__()
        self.center = center.copy()
        self.circulation = circulation

    def value(self, t, x):
        r = np.linalg.norm(x - self.center)
        e_theta = np.array([-(x - self.center)[1] / r,
                            (x - self.center)[0] / r])
        return self.circulation / (2 * np.pi * r) * e_theta

    def d_value(self, t, x):
        r = np.linalg.norm(x - self.center)
        return self.circulation / (2 * np.pi * r ** 4) * np.array(
            [[2 * (x[0] - self.center[0]) * (x[1] - self.center[1]),
              (x[1] - self.center[1]) ** 2 - (x[0] - self.center[0]) ** 2],
             [(x[1] - self.center[1]) ** 2 - (x[0] - self.center[0]) ** 2,
              -2 * (x[0] - self.center[0]) * (x[1] - self.center[1])]])


class RankineVortexFF(FlowField):

    def __init__(self, center: Union[float, ndarray],
                 circulation: Union[float, ndarray], radius: Union[float, ndarray], t_start=0., t_end=None):
        """
        A vortex from potential theory
        :param center: Coordinates of the vortex center. Shape (2,) if steady else shape (nt, 2)
        :param circulation: Circulation of the vortex. Positive is ccw vortex. Scalar for steady vortex
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

        self.center = np.array(center)
        self.circulation = np.array(circulation)
        self.radius = np.array(radius)
        if nt_int is not None and not (self.center.shape[0] == self.circulation.shape[0] == self.radius.shape[0]):
            print('Incoherent sizes', file=sys.stderr)
            exit(1)
        self.zero_ceil = 1e-3

    def max_speed(self, t=0.):
        _, gamma, radius = self.params(t)
        return gamma / (radius * 2 * np.pi)

    def params(self, t):
        if self.t_end is None:
            return self.center, self.circulation, self.radius
        i, alpha = self._index(t)
        omega = (1 - alpha) * self.center[i] + alpha * self.center[i + 1]
        gamma = (1 - alpha) * self.circulation[i] + alpha * self.circulation[i + 1]
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


class StateLinearFF(FlowField):
    """
    Linear variation with state variable
    """

    def __init__(self, gradient: ndarray, origin: ndarray, value_origin: ndarray):
        """
        Flow field value = gradient @ (state - origin) + value_origin
        :param gradient: The flow field gradient
        :param origin: The origin point for the origin value
        :param value_origin: The origin value
        """
        super().__init__()

        self.gradient: ndarray = gradient.copy()
        self.origin: ndarray = origin.copy()
        self.value_origin: ndarray = value_origin.copy()

    def value(self, _, x):
        return self.gradient.dot(x - self.origin) + self.value_origin

    def d_value(self, _, x):
        return self.gradient


class LinearFFT(FlowField):
    """
    Linear variation with state variable and linear evolution of spatial gradient
    """

    def __init__(self, gradient_diff_time: ndarray, gradient_init: ndarray,
                 origin: ndarray, value_origin: ndarray):
        """
        Flow field value = (gradient_diff_time * t + gradient_init) @ (state - origin) + value_origin
        :param gradient_diff_time: Derivative of spatial gradient wrt time, shape (2, 2)
        :param gradient_init: Initial spatial gradient, shape (2, 2)
        :param origin: The origin point for the origin value, shape (2,)
        :param value_origin: The origin value, shape (2,)
        """
        super().__init__()

        self.gradient_diff_time = gradient_diff_time.copy()
        self.gradient_init = gradient_init.copy()
        self.origin = origin.copy()
        self.value_origin = value_origin.copy()

    def value(self, t, x):
        (self.gradient_diff_time * t + self.gradient_init) @ (x - self.origin) + self.value_origin

    def d_value(self, t, x):
        return self.gradient_diff_time * t + self.gradient_init


class PointSymFF(FlowField):
    """
    From Techy 2011 (DOI 10.1007/s11370-011-0092-9)
    """

    def __init__(self, center: Union[ndarray | tuple[float, float]], gamma: float, omega: float):
        """
        :param center: Center coordinates
        :param gamma: Divergence
        :param omega: Curl
        """
        super().__init__()

        self.center = center.copy() if isinstance(center, ndarray) else np.array(center)

        self.mat = np.array([[gamma, -omega], [omega, gamma]])

    def value(self, t, x):
        return self.mat @ (x - self.center)

    def d_value(self, t, x):
        return self.mat


class GyreFF(FlowField):

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


class DoubleGyreDampedFF(FlowField):

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
        self.double_gyre = GyreFF(x_center, y_center, x_wl, y_wl, ampl)
        self.center_damp = np.array((x_center_damp, y_center_damp))
        self.lambda_x_damp = lambda_x_damp
        self.lambda_y_damp = lambda_y_damp

    def value(self, t, x):
        xx = np.diag((1 / self.lambda_x_damp, 1 / self.lambda_y_damp)) @ (x - self.center_damp)
        damp = 1 / (1 + xx[0] ** 2 + xx[1] ** 2)
        return self.double_gyre.value(t, x) * damp


class RadialGaussFF(FlowField):
    def __init__(self, center: ndarray, radius: float, sdev: float, v_max: float):
        """
        :param center: Center
        :param radius: Radial distance of maximum flow field value (absolute value)
        :param sdev: Gaussian standard deviation
        :param v_max: Maximum flow field value
        """
        super().__init__()

        self.center = center.copy()

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


class RadialGaussFFT(FlowField):
    """
    Time-varying version of radial Gauss flow field
    """

    def __init__(self, center, radius, sdev, v_max, t_end, t_start=0.):
        """
        :param center: ndarray of size nt x 2
        :param radius: ndarray of size nt
        :param sdev: ndarray of size nt
        :param v_max: ndarray of size nt
        :param t_end: end of time window. Flow field is supposed to be regularly sampled in the time window
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


class BandGaussFF(FlowField):
    """
    Linear band of gaussian flow field
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


class BandFF(FlowField):
    """
    Band of flow field. NON DIFFERENTIABLE (should be instanciated as discrete flow field)
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
        raise NotImplementedError('Please discretize FF before using derivatives')


class TrapFF(FlowField):
    def __init__(self, ff_val: ndarray, center: ndarray, radius: ndarray,
                 rel_wid=0.05, t_start=0., t_end: Optional[float] = None):
        """
        :param ff_val: (nt,)
        :param center: (nt, 2)
        :param radius: (nt,)
        """
        if ff_val.shape[0] != radius.shape[0]:
            raise Exception('Incoherent shapes')
        super().__init__(t_start=t_start, t_end=t_end, nt_int=ff_val.shape[0])
        self.ff_val = ff_val.copy()
        self.center = center.copy()
        self.radius = radius.copy()
        # Relative obstacle variation width
        self.rel_wid = rel_wid

    @staticmethod
    def sigmoid(r, wid):
        return 1 / (1 + np.exp(-4 / wid * r))

    @staticmethod
    def d_sigmoid(r, wid):
        lam = 4 / wid
        s = TrapFF.sigmoid(r, wid)
        return lam * s * (1 - s)

    def value(self, t, x):
        i, alpha = self._index(t)
        ff_val = (1 - alpha) * self.ff_val[i] + (0. if alpha < 1e-3 else alpha * self.ff_val[i + 1])
        center = (1 - alpha) * self.center[i] + (0. if alpha < 1e-3 else alpha * self.center[i + 1])
        radius = (1 - alpha) * self.radius[i] + (0. if alpha < 1e-3 else alpha * self.radius[i + 1])
        if (x - center) @ (x - center) < 1e-8 * radius ** 2:
            return np.zeros(2)
        else:
            r = np.linalg.norm((x - center))
            e_r = (x - center) / r
            return -ff_val * TrapFF.sigmoid(r - radius, radius * self.rel_wid) * e_r

    def d_value(self, t, x):
        i, alpha = self._index(t)
        ff_val = (1 - alpha) * self.ff_val[i] + (0. if alpha < 1e-3 else alpha * self.ff_val[i + 1])
        center = (1 - alpha) * self.center[i] + (0. if alpha < 1e-3 else alpha * self.center[i + 1])
        radius = (1 - alpha) * self.radius[i] + (0. if alpha < 1e-3 else alpha * self.radius[i + 1])
        if (x - center) @ (x - center) < 1e-8 * radius ** 2:
            return np.zeros((2, 2))
        else:
            r = np.linalg.norm((x - center))
            e_r = (x - center) / r
            P = np.array(((e_r[0], e_r[1]), (-e_r[1] / r, e_r[0] / r)))
            return -ff_val * (P.transpose() @ np.diag((TrapFF.d_sigmoid(r - radius, radius * self.rel_wid),
                                                       TrapFF.sigmoid(r - radius, radius * self.rel_wid))) @ P)


class ChertovskihFF(FlowField):

    def __init__(self):
        super().__init__()

    def value(self, t, x):
        return np.array((x[0] * (x[1] + 3) / 4, -x[0] * x[0]))

    def d_value(self, t, x):
        return np.array((((x[1] + 3) / 4, x[0] / 4),
                         (-2 * x[0], 0)))
