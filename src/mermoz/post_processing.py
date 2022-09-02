import os
import sys
import h5py
import json
import scipy.interpolate as itp
from numpy import sin, pi, cos
from math import atan2, asin
import matplotlib.pyplot as plt

sys.path.extend('/home/bastien/Documents/work/mdisplay/src/mdisplay/')
from mdisplay.misc import windy_cm

from mermoz.wind import DiscreteWind
from mermoz.misc import *

path_colors = ['b', 'g', 'r', 'c', 'm', 'y']


class TrajStats:

    def __init__(self, length: float, duration: float, gs: ndarray, crosswind: ndarray, tgwind: ndarray,
                 controls: ndarray):
        self.length = length
        self.duration = duration
        self.gs = np.zeros(gs.shape)
        self.gs[:] = gs
        self.cw = np.zeros(crosswind.shape)
        self.cw[:] = crosswind
        self.tw = np.zeros(tgwind.shape)
        self.tw[:] = tgwind
        self.controls = np.zeros(controls.shape)
        self.controls[:] = controls


class PostProcessing:

    def __init__(self, output_dir, traj_fn=None, wind_fn=None, param_fn=None):
        self.output_dir = output_dir
        self.traj_fn = traj_fn if traj_fn is not None else 'trajectories.h5'
        self.wind_fn = wind_fn if wind_fn is not None else 'wind.h5'
        self.param_fn = param_fn if param_fn is not None else 'params.json'
        with open(os.path.join(self.output_dir, self.param_fn), 'r') as f:
            pd = json.load(f)
            try:
                self.coords = pd['coords']
            except KeyError:
                print('[postproc] Missing coordinates type', file=sys.stderr)
                exit(1)

            success = False
            for name in ['va', 'airspeed']:
                try:
                    self.va = pd[name]
                    success = True
                except KeyError:
                    pass
            if not success:
                print('[postproc] Airspeed not found in parameters, switching to default value')
                self.va = AIRSPEED_DEFAULT

        wind_fp = os.path.join(self.output_dir, self.wind_fn)
        self.wind = DiscreteWind(interp='linear')
        self.wind.load(wind_fp)

        self.trajs = []

    def load(self, opti_only=False):
        traj_fp = os.path.join(self.output_dir, self.traj_fn)
        f = h5py.File(traj_fp, "r")
        for k, traj in enumerate(f.values()):
            if not opti_only or traj.attrs['type'] in ['integral', 'optimal']:
                print(traj.attrs['type'], end=' ')
                try:
                    print(traj.attrs['info'])
                except KeyError:
                    print()
                _traj = {}
                _traj['data'] = np.zeros(traj['data'].shape)
                _traj['data'][:] = traj['data']
                _traj['controls'] = np.zeros(traj['controls'].shape)
                _traj['controls'][:] = traj['controls']
                _traj['ts'] = np.zeros(traj['ts'].shape)
                _traj['ts'][:] = traj['ts']

                _traj['type'] = traj.attrs['type']
                _traj['last_index'] = traj.attrs['last_index']
                _traj['interrupted'] = traj.attrs['interrupted']
                _traj['coords'] = traj.attrs['coords']
                _traj['label'] = traj.attrs['label']
                _traj['info'] = traj.attrs['info']
                self.trajs.append(_traj)
        f.close()

    def stats(self, fancynormplot=False, only_opti=False):
        fig, ax = plt.subplots(ncols=3, nrows=2)
        decorate(ax[0, 0], 'Delay per length unit', 'Point', '[s/m]', ylim=(0., 3 / self.va))
        decorate(ax[0, 1], 'Crosswind', 'Point', '[m/s]', ylim=(-1. * self.va, 1. * self.va))
        decorate(ax[0, 2], 'Tangent wind', 'Point', '[m/s]', ylim=(-1. * self.va, 1. * self.va))
        decorate(ax[1, 0], 'Ground speed', 'Point', '[m/s]', ylim=(0., 2. * self.va))
        decorate(ax[1, 1], 'Wind norm', 'Point', '[m/s]', ylim=(0, 1.1 * self.va))
        for k, traj in enumerate(self.trajs):
            points = np.zeros(traj['data'].shape)
            points[:] = traj['data']
            nt = traj['last_index']
            color = path_colors[traj['label'] % len(path_colors)]
            tstats = self.point_stats(points, last_index=nt)
            x = np.linspace(0, 1., nt - 1)
            ax[0, 0].plot(x, 1 / tstats.gs, label=f'{k}' + ('_opt' if traj['type'] == 'optimal' else ''), color=color)
            ax[0, 1].plot(x, tstats.cw, color=color)
            ax[0, 2].plot(x, tstats.tw, color=color)
            y = np.sqrt(tstats.cw ** 2 + tstats.tw ** 2)
            if fancynormplot:
                xx = np.linspace(0, nt - 1, 10 * (nt - 1))
                yy = itp.interp1d(x, y)(xx)
                ax[1, 1].scatter(xx, yy, c=windy_cm(yy / windy_cm.norm_max), s=0.5, color=color)
            else:
                ax[1, 1].plot(x, y, color=color)
            ax[1, 0].plot(x, tstats.gs)
            print(f'{k} : {tstats.duration / 3600:.2f}h, {tstats.length / 1000:.2f}km')
        ax[0, 0].legend()
        plt.show()

    def point_stats(self, points, last_index=-1):
        """
        Return ground speed, crosswind and tangent wind for each node of the trajectory
        :param points: ndarray (n, 2) of trajectory points
        :return: (ground speed norm, crosswind, tgwind, duration) three ndarray (n-1,) and a float
        """
        if last_index < 0:
            n = points.shape[0]
        else:
            n = last_index
        # zero_ceil = np.mean(np.linalg.norm(points, axis=1)) * 1e-9
        gs = np.zeros(n - 1)
        cw = np.zeros(n - 1)
        tw = np.zeros(n - 1)
        controls = np.zeros(n - 1)
        duration = 0.
        length = float(np.sum([distance(points[i], points[i+1], self.coords) for i in range(n-1)]))
        for i in range(n - 1):
            p = np.zeros(2)
            p2 = np.zeros(2)
            p[:] = points[i]
            p2[:] = points[i + 1]
            delta_x = p2 - p
            dx_norm = np.linalg.norm(delta_x)
            e_dx = delta_x / np.linalg.norm(delta_x)
            dx_arg = atan2(delta_x[1], delta_x[0])
            w = self.wind.value(0., p)
            w_norm = np.linalg.norm(w)
            w_arg = atan2(w[1], w[0])
            right = w_norm / self.va * sin(w_arg - dx_arg)

            def gs_f(uu, va, w, e_dx):
                return (np.array((cos(uu), sin(uu))) * va + w) @ e_dx

            u = max([dx_arg - asin(right), dx_arg + asin(right) - pi], key=lambda uu: gs_f(uu, self.va, w, e_dx))

            gs[i] = gs_v = gs_f(u, self.va, w, e_dx)
            cw[i] = np.cross(e_dx, w)
            tw[i] = e_dx @ w
            controls[i] = u
            duration += dx_norm / gs_v

        tstats = TrajStats(length, duration, gs, cw, tw, controls)
        return tstats


if __name__ == '__main__':
    pp = PostProcessing('/home/bastien/Documents/work/mermoz/output/example_solver-pa_linear_0')
    pp.stats(only_opti=True)
