import os

import h5py
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


class DisplayPlotly:

    def __init__(self, dir_path):
        self.output_dir = dir_path
        self.trajs_fname = 'trajectories.h5'
        self.tl_traj = 0.
        self.tu_traj = 1.
        self.trajs = []
        self.fronts = []

    def load(self):
        trajfiles = [name for name in os.listdir(self.output_dir)
                     if name.startswith(self.trajs_fname.split('.')[0])]
        trajs_fpath = list(map(lambda fn: os.path.join(self.output_dir, fn), trajfiles))
        if len(trajs_fpath) == 0 or not os.path.exists(trajs_fpath[0]):
            return
        max_len = 0
        for traj_fpath in trajs_fpath:
            with h5py.File(traj_fpath, 'r') as f:
                for k, traj in enumerate(f.values()):
                    nt = traj['ts'].shape[0]
                    if nt > max_len:
                        max_len = nt
                        self.tl_traj = np.min(traj['ts'])
                        self.tu_traj = np.max(traj['ts'])
        self.fronts = [[]] * (max_len + 1)

        def time_index(t):
            return int((t - self.tl_traj) / (self.tu_traj - self.tl_traj) * max_len)

        for traj_fpath in trajs_fpath:
            with h5py.File(traj_fpath, 'r') as f:
                for k, traj in tqdm(enumerate(f.values())):

                    # if 'info' in traj.attrs.keys() and traj.attrs['info'] in self.traj_filter:
                    #     continue
                    # if traj['ts'].shape[0] <= 1:
                    #     continue
                    # if traj.attrs['coords'] != self.coords:
                    #     print(
                    #         f'[Warning] Traj. coord type {traj.attrs["coords"]} differs from display mode {self.coords}')

                    """
                    if self.nt_tick is not None:
                        kwargs['nt_tick'] = self.nt_tick
                    """
                    _traj = {}
                    _traj['data'] = np.zeros(traj['data'].shape)
                    _traj['data'][:] = traj['data']
                    _traj['controls'] = np.zeros(traj['controls'].shape)
                    _traj['controls'][:] = traj['controls']
                    _traj['ts'] = np.zeros(traj['ts'].shape)
                    _traj['ts'][:] = traj['ts']
                    if _traj['ts'].shape[0] == 0:
                        continue

                    # if 'airspeed' in traj.keys():
                    #     _traj['airspeed'] = np.zeros(traj['airspeed'].shape)
                    #     _traj['airspeed'][:] = traj['airspeed']

                    if 'energy' in traj.keys() and traj['energy'].shape[0] > 0:
                        _traj['energy'] = np.zeros(traj['energy'].shape)
                        _traj['energy'][:] = traj['energy']
                        # cmin = _traj['energy'][:].min()
                        # cmax = _traj['energy'][:].max()
                        # if self.engy_min is None or cmin < self.engy_min:
                        #     self.engy_min = cmin
                        # if self.engy_max is None or cmax > self.engy_max:
                        #     self.engy_max = cmax

                    _traj['type'] = traj.attrs['type']
                    li = _traj['last_index'] = traj.attrs['last_index']
                    _traj['interrupted'] = traj.attrs['interrupted']
                    _traj['coords'] = traj.attrs['coords']
                    _traj['label'] = traj.attrs['label']
                    # Backward compatibility
                    if 'info' in traj.attrs.keys():
                        _traj['info'] = traj.attrs['info']
                    else:
                        _traj['info'] = ''

                    # Label trajectories belonging to extremal fields
                    # if _traj['info'].startswith('ef'):
                    #     ef_id = int(_traj['info'].strip().split('_')[1])
                    #     if ef_id not in self.ef_ids:
                    #         self.ef_ids.append(ef_id)
                    #         self.ef_trajgroups[ef_id] = []
                    #     self.ef_trajgroups[ef_id].append(_traj)

                    if 'energy' in traj.keys():
                        for it, t in enumerate(traj['ts']):
                            a = self.fronts[time_index(t)]
                            try:
                                self.fronts[time_index(t)].append(np.array(tuple(traj['data'][it]) + (traj['energy'][it],)))
                            except IndexError:
                                pass

                    self.trajs.append(_traj)

        fronts = []
        for fr in self.fronts:
            fronts.append(np.array(fr))
        self.fronts = fronts

    def run(self):
        fig = go.Figure()
        i = 0
        for fr in self.fronts:
            if i > 10:
                break
            fig.add_trace(go.Scatter3d(x=fr[:, 0], y=fr[:, 1], z=fr[:, 2]))
            i += 1
        fig.show()


if __name__ == '__main__':
    disp = DisplayPlotly('/home/bastien/Documents/work/dabry/output/movor')
    disp.load()
    disp.run()
