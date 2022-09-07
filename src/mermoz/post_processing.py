import os
import sys
import h5py
import json

import numpy as np
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

    def __init__(self,
                 length: float,
                 duration: float,
                 gs: ndarray,
                 crosswind: ndarray,
                 tgwind: ndarray,
                 vas: ndarray,
                 controls: ndarray):
        self.length = length
        self.duration = duration
        self.gs = np.zeros(gs.shape)
        self.gs[:] = gs
        self.cw = np.zeros(crosswind.shape)
        self.cw[:] = crosswind
        self.tw = np.zeros(tgwind.shape)
        self.tw[:] = tgwind
        self.vas = np.zeros(vas.shape)
        self.vas[:] = vas
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
                try:
                    _traj['info'] = traj.attrs['info']
                except KeyError:
                    # Backward compatibility
                    _traj['info'] = ""
                self.trajs.append(_traj)
        f.close()

    def stats(self, fancynormplot=False, only_opti=False):
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
        decorate(ax[0, 0], 'Delay per length unit', 'Time (scaled)', '[s/m]', ylim=(0., 3 / self.va))
        decorate(ax[0, 1], 'Crosswind', 'Time (scaled)', '[m/s]', ylim=(-1. * self.va, 1. * self.va))
        decorate(ax[0, 2], 'Tangent wind', 'Time (scaled)', '[m/s]', ylim=(-1. * self.va, 1. * self.va))
        decorate(ax[1, 0], 'Ground speed', 'Time (scaled)', '[m/s]', ylim=(0., 2. * self.va))
        decorate(ax[1, 1], 'Wind norm', 'Time (scaled)', '[m/s]', ylim=(0, 1.1 * self.va))
        ax[0, 2].axhline(0., color='black')
        ax[0, 1].axhline(0., color='black')
        tl, tu = None, None
        got_tu = False
        for k, traj in enumerate(self.trajs):
            ttl = traj['ts'][0]
            ttu = traj['ts'][traj['last_index']]
            if tl is None or ttl < tl:
                tl = ttl
            if not got_tu:
                if traj['type'] == 'optimal' and traj['info'].startswith('ef'):
                    tu = ttu
                    got_tu = True
                elif traj['type'] == 'optimal':
                    # RFT traj
                    tu = ttu
                    got_tu = True
            if not got_tu:
                if tu is None or ttu > tu:
                    tu = ttu
        for a in ax:
            for b in a:
                b.set_xlim(tl, tu)
        for k, traj in enumerate(self.trajs):
            points = np.zeros(traj['data'].shape)
            points[:] = traj['data']
            ts = np.array(traj['ts'])
            nt = traj['last_index']
            color = path_colors[k % len(path_colors)]
            tstats = self.point_stats(ts, points, last_index=nt)
            x = np.linspace(0, 1., nt - 1)
            ax[0, 0].plot(ts[:nt-1], 1 / tstats.gs, label=f'{traj["info"]}')
            ax[0, 1].plot(ts[:nt-1], tstats.cw, color=color)
            ax[0, 2].plot(ts[:nt-1], tstats.tw, color=color)
            y = np.sqrt(tstats.cw ** 2 + tstats.tw ** 2)
            if fancynormplot:
                xx = np.linspace(0, nt - 1, 10 * (nt - 1))
                yy = itp.interp1d(x, y)(xx)
                ax[1, 1].scatter(xx, yy, c=windy_cm(yy / windy_cm.norm_max), s=0.5, color=color)
            else:
                ax[1, 1].plot(ts[:nt-1], y, color=color)
            ax[1, 0].plot(ts[:nt-1], tstats.gs)
            hours = int(tstats.duration / 3600)
            minutes = int((tstats.duration - 3600*hours) / 60.)
            print(
                f'{k} : ' +
                f'{traj["type"] + " " + traj["info"]:<25} ' +
                f'{hours}h{minutes}m, ' +
                f'{tstats.length / 1000:.2f}km, ' +
                f'mean gs {np.mean(tstats.gs):.2f} m/s, ' +
                f'mean airspeed {np.mean(tstats.vas):.2f} m/s'
            )
        ax[0, 0].legend(loc='center left', bbox_to_anchor=(-1., 0.5))
        plt.show()

    def point_stats(self, ts, points, last_index=-1):
        """
        Return ground speed, crosswind and tangent wind for each node of the trajectory
        :param ts: ndarray (n,) of timestamps
        :param points: ndarray (n, 2) of trajectory points
        :return: a TrajStats object
        """
        if last_index < 0:
            n = points.shape[0]
        else:
            n = last_index
        # zero_ceil = np.mean(np.linalg.norm(points, axis=1)) * 1e-9
        gs = np.zeros(n - 1)
        cw = np.zeros(n - 1)
        tw = np.zeros(n - 1)
        vas = np.zeros(n - 1)
        controls = np.zeros(n - 1)
        duration = 0.
        t = ts[0]
        length = float(np.sum([distance(points[i], points[i + 1], self.coords) for i in range(n - 1)]))
        for i in range(n - 1):
            dt = ts[i + 1] - ts[i]
            p = np.zeros(2)
            p2 = np.zeros(2)
            p[:] = points[i]
            p2[:] = points[i + 1]
            corr_mat = EARTH_RADIUS * np.diag((np.cos(p[1]), 1.)) if self.coords == COORD_GCS else np.diag((1., 1.))
            delta_x = (p2 - p)
            dx_norm = np.linalg.norm(delta_x)
            e_dx = delta_x / dx_norm
            dx_arg = atan2(delta_x[1], delta_x[0])
            w = self.wind.value(t, p)
            w_norm = np.linalg.norm(w)
            w_arg = atan2(w[1], w[0])
            right = w_norm / self.va * sin(w_arg - dx_arg)
            gsv = corr_mat @ delta_x / dt
            u = atan2(*((gsv - w)[::-1]))
            v_a = np.linalg.norm(gsv - w)

            # def gs_f(uu, va, w, e_dx):
            #     return (np.array((cos(uu), sin(uu))) * va + w) @ e_dx
            #
            # u = max([dx_arg - asin(right), dx_arg + asin(right) - pi], key=lambda uu: gs_f(uu, self.va, w, e_dx))

            #gs[i] = gs_v = gs_f(u, self.va, w, e_dx)
            gs[i] = np.linalg.norm(gsv)
            cw[i] = np.cross(e_dx, w)
            tw[i] = e_dx @ w
            vas[i] = v_a
            controls[i] = u if self.coords == COORD_CARTESIAN else np.pi/2. - u
            t = ts[i + 1]
            duration += dt

        tstats = TrajStats(length, duration, gs, cw, tw, vas, controls)
        return tstats


if __name__ == '__main__':
    pp = PostProcessing('/home/bastien/Documents/work/mermoz/output/example_solver-pa_linear_0')
    pp.stats(only_opti=True)
