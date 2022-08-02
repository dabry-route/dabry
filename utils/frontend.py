import argparse
import os.path
import sys
import numpy as np
import easygui
import h5py
import json

sys.path.extend(['/home/bastien/Documents/work/mermoz',
                 '/home/bastien/Documents/work/mermoz/src',
                 '/home/bastien/Documents/work/mdisplay',
                 '/home/bastien/Documents/work/mdisplay/src'])
from mdisplay.display import Display
from mermoz.misc import *


class FrontendHandler:

    def __init__(self, mode='notebook'):
        self.display = None
        self.case_name = None
        self.dd = None
        # Default directory containing mermoz output files
        self.output_dir = '/home/bastien/Documents/work/mermoz/output'
        self.output_path = None
        self.mode = mode
        self.pp_params = {}

    def configure(self, opti_only=False):
        if self.case_name == 'example_geodesic':
            self.display.set_title('Geodesic approximation with extremals')
            bl = np.array([-76., 30.])
            tr = np.array([5., 60.])
            self.display.load_params()
            self.display.set_coords('gcs')
            self.display.setup(bl=bl, tr=tr)
            self.display.draw_trajs(nolabels=True)
            lon_ny, lat_ny = self.display.geodata.get_coords('New York')
            lon_par, lat_par = self.display.geodata.get_coords('Paris')
            self.display.map.drawgreatcircle(lon_ny, lat_ny, lon_par, lat_par, linewidth=1, color='b', alpha=0.4,
                                             linestyle='--',
                                             label='Great circle', zorder=4)
            self.display.draw_point_by_name('New York')
            self.display.draw_point_by_name('Paris')

        elif self.case_name == 'example_solver_linearwind' or self.case_name == 'example_ft_linearwind':
            self.display.set_coords('cartesian')
            self.display.set_title('Analytic solution comparison')
            self.display.setup()
            self.display.load_params()
            self.display.draw_wind()
            self.display.draw_trajs()
            self.display.draw_point(1e6, 0., label='Target')
            if self.case_name == 'example_ft_linearwind':
                self.display.draw_rff()

        elif self.case_name == 'XXXexample_front_tracking':
            self.display.set_coords('gcs')
            self.display.set_title('Front tracking example')

            self.display.setup(projection='lcc')

            self.display.load_params()
            self.display.draw_wind()
            self.display.draw_rff(debug=False)
            self.display.draw_trajs(nolabels=True)
            self.display.draw_point_by_name('Natal')

        elif self.case_name == 'XXXexample_front_tracking2' or self.case_name == 'example_front_tracking_linearinterp':
            self.display.set_coords('gcs')
            self.display.set_title('Front tracking example')

            self.display.setup(projection='lcc')

            self.display.load_params()
            self.display.draw_wind()
            self.display.draw_rff()
            self.display.draw_trajs(nolabels=True)

            self.display.draw_point_by_name('Tripoli')
            self.display.draw_point_by_name('Athenes')

        elif self.case_name == 'example_front_tracking3':
            self.display.set_coords('gcs')
            self.display.set_title('Front tracking example')

            self.display.setup(projection='lcc')

            self.display.load_params()
            self.display.draw_wind()
            self.display.draw_rff()
            self.display.draw_trajs(nolabels=True)

            self.display.draw_point_by_name('New York')
            self.display.draw_point_by_name('Paris')

        elif self.case_name == 'example_front_tracking4':
            self.display.set_coords('gcs')
            self.display.set_title('Front tracking example')

            self.display.setup()

            self.display.load_params()
            self.display.draw_wind()
            self.display.draw_rff()
            self.display.draw_trajs(nolabels=True)

        elif self.case_name == 'example_solver':
            self.display.nocontrols = True
            self.display.set_title('Solver test')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=True)
            self.display.draw_solver()
            # self.display.legend()

        elif self.case_name == 'example_ft_vor3':
            self.display.nocontrols = True
            self.display.set_coords('cartesian')
            self.display.set_title('Front tracking on Rankine vortices')
            self.display.setup()
            self.display.load_params()
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=True, filename='../example_solver/trajectories.h5')
            self.display.draw_rff()

        elif self.case_name == 'example_solver_dn':
            self.display.nocontrols = True
            self.display.set_title('Solver')
            self.display.load_params()
            self.display.setup(projection='lcc')
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=False, opti_only=True)
            self.display.draw_solver(labeling=False)
            # lon_ny, lat_ny = self.display.geodata.get_coords('New York')
            # lon_par, lat_par = self.display.geodata.get_coords('Paris')
            # self.display.map.drawgreatcircle(lon_ny, lat_ny, lon_par, lat_par, linewidth=1, color='b', alpha=0.4,
            #                                  linestyle='--',
            #                                  label='Great circle', zorder=4)
        elif self.case_name == 'example_solver_test':
            self.display.nocontrols = True
            self.display.set_title('Solver')
            self.display.load_params()
            self.display.setup(projection='ortho')
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=False)
            self.display.draw_solver()
            lon_ny, lat_ny = self.display.geodata.get_coords('New York')
            lon_par, lat_par = self.display.geodata.get_coords('Paris')
            self.display.map.drawgreatcircle(lon_ny, lat_ny, lon_par, lat_par, linewidth=1, color='b', alpha=0.4,
                                             linestyle='--',
                                             label='Great circle', zorder=4)
        elif 'double-gyre-ku2016' in self.case_name or 'double-gyre-li2020' in self.case_name:
            self.display.nocontrols = True
            self.display.set_title(f'{self.case_name.split("-")[-1]}')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind(autoscale=True, wind_nointerp=True)
            self.display.draw_trajs(nolabels=True, opti_only=opti_only)
            self.display.draw_rff()
            self.display.draw_solver()

            # def on_plot_hover(event):
            #     # Iterating over each data member plotted
            #     for curve in self.display.mainax.get_lines():
            #         # Searching which data member corresponds to current mouse position
            #         if curve.contains(event)[0]:
            #             with h5py.File(self.display.trajs_fpath, 'r') as f:
            #                 value = RAD_TO_DEG * f[str(curve.get_gid())]["controls"][0]
            #                 self.display.ax_info.set_text(str(value))
            #                 print(f'{value}')
            #             break
            #
            # self.display.mainfig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

        elif '3obs' in self.case_name:
            self.display.nocontrols = True
            self.display.set_title('3obs')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=False, opti_only=False)
            self.display.draw_solver()
            self.display.draw_rff()
        elif 'big_rankine' in self.case_name:
            self.display.nocontrols = True
            self.display.set_title('big-rankine')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind(wind_nointerp=True)
            self.display.draw_trajs(nolabels=False, opti_only=False)
            self.display.draw_solver()
            self.display.draw_rff(slice=(0, 5))

        elif self.case_name in ['example_test_grib']:
            self.display.nocontrols = True
            self.display.set_title('Test grib')
            self.display.load_params()
            self.display.setup(projection='ortho')
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=False, opti_only=False)
            # self.display.draw_solver()
            # self.display.draw_rff()

        elif self.case_name in ['example_test_flatten']:
            self.display.nocontrols = True
            self.display.set_title('Test flatten')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind(wind_nointerp=True)
            # self.display.draw_trajs(nolabels=False, opti_only=False)
            # self.display.draw_solver()
            self.display.draw_rff()

        elif self.case_name in ['example_solver_dakar-natal']:
            self.display.nocontrols = True
            self.display.set_title('Optimal trajectory between Dakar and Natal')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=False, opti_only=False)
            self.display.draw_solver(labeling=False)
            # self.display.draw_rff()

        elif self.case_name in ['example_test_flatten_ref']:
            self.display.nocontrols = True
            self.display.set_title('Test flatten')
            self.display.load_params()
            self.display.setup(projection='ortho')
            self.display.draw_wind(wind_nointerp=True)
            self.display.draw_trajs(nolabels=False, opti_only=False)
            # self.display.draw_solver()
            self.display.draw_rff()

        elif self.case_name in ['example_solver_san-juan_dublin_ortho', 'example_solver-rp_sanjuan-dublin-ortho']:
            self.display.nocontrols = True
            self.display.set_title('Test flatten')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind(wind_nointerp=True)
            self.display.draw_trajs(nolabels=False, opti_only=False)
            # self.display.draw_solver()
            self.display.draw_rff(slice=(1, 10))

        elif 'solver' in self.case_name:
            print(f'Using default solver setup script for unknown case "{self.case_name}"', file=sys.stderr)
            self.display.nocontrols = True
            self.display.set_title(os.path.basename(self.output_path))
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=False)
            self.display.draw_rff()
            self.display.draw_solver()
        elif 'wf' in self.case_name:
            print(f'Using default wind field setup script for unknown case "{self.case_name}"', file=sys.stderr)
            self.display.set_title(os.path.basename(self.output_path))
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind()
        else:
            print(f'Using default setup script for unknown case "{self.case_name}"', file=sys.stderr)
            self.display.nocontrols = True
            self.display.set_title(os.path.basename(self.output_path))
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=True)
            self.display.draw_rff()
            self.display.draw_solver()

    def select_example(self, *args, latest=False):
        if self.mode == 'user':
            self.output_path = args[0]
            self.case_name = os.path.basename(self.output_path)
        elif self.mode == 'notebook':
            from ipywidgets import Dropdown
            from IPython.core.display import display
            example_path = os.path.join('..', 'output')
            cache_fp = os.path.join(example_path, '.frontend_cache.txt')
            default_value = None
            if os.path.exists(cache_fp):
                with open(cache_fp, 'r') as f:
                    default_value = f.readline()
            option_list = sorted(os.listdir(example_path))
            new_ol = []
            for e in option_list:
                if not e.startswith('example_'):
                    del e
                else:
                    new_ol.append(e.split('example_')[1])
            kwargs = {'description': "Choose one:",
                      'options': new_ol}
            if default_value is not None and os.path.exists(os.path.join(example_path, default_value)):
                kwargs['value'] = default_value
            self.dd = Dropdown(**kwargs)

            def handler(change):
                try:
                    with open(cache_fp, 'w') as f:
                        f.writelines(change['new'])
                except KeyError:
                    pass

            self.dd.observe(handler, names='value')
            display(self.dd)
        else:
            class ns:
                def __init__(self, value):
                    self.__dict__.update(value=value)

            cache_fp = os.path.join('..', 'output', '.frontend_cache2.txt')
            last = '{LAST}'
            latest = '{LATEST}'
            nlist = [dd for dd in os.listdir(self.output_dir) if
                     os.path.isdir(os.path.join(self.output_dir, dd)) and not dd.startswith('.')]
            name = easygui.choicebox('Choose example', 'Example1', [latest, last] + list(sorted(nlist)))
            if name is None:
                print('No example selected, quitting', file=sys.stderr)
                exit(1)
            sel_dir = os.path.join(self.output_dir, name)
            # sel_dir = easygui.diropenbox(default=os.path.join('..', 'output'))
            if name == last:
                print('Opening last file')
                with open(cache_fp, 'r') as f:
                    sel_dir = f.readline()
                    print(sel_dir)
                self.dd = ns(os.path.basename(sel_dir))
            elif name == latest:
                print('Opening latest file')
                all_subdirs = list(map(lambda n: os.path.join(self.output_dir, n), nlist))
                all_params = []
                for dd in all_subdirs:
                    params_path = os.path.join(dd, 'params.json')
                    if os.path.exists(params_path):
                        all_params.append(params_path)

                latest_subdir = os.path.dirname(max(all_params, key=os.path.getmtime))
                print(latest_subdir)
                self.dd = ns(os.path.basename(latest_subdir))

            else:
                # self.dd = ns(os.path.basename(sel_dir).split('example_')[1])
                self.dd = ns(os.path.basename(sel_dir))
                with open(cache_fp, 'w') as f:
                    f.writelines(sel_dir)

    def run_frontend(self, ex_name=None, noparams=True, opti_only=False, noshow=False):
        # if self.mode == 'default':
        #     if ex_name is None:
        #         self.output_path = easygui.diropenbox(default=os.path.join('..', 'output'))
        #         if self.output_path is None:
        #             exit(0)
        #     else:
        #         self.output_path = os.path.join('..', 'output', ex_name)
        #     self.case_name = os.path.basename(self.output_path)
        if self.mode in ['notebook', 'default']:
            self.output_path = os.path.join('..', 'output', self.example_name())
            self.case_name = os.path.basename(self.output_path)
        elif self.mode == 'user':
            pass
        else:
            print(f'Unknown mode {self.mode}', file=sys.stderr)
            exit(1)

        self.display = Display()
        self.display.set_output_path(self.output_path)
        self.configure(opti_only=opti_only)
        self.display.update_title()
        if not noshow:
            self.display.show(noparams=noparams)

    def show_params(self):
        from IPython.core.display import HTML
        return HTML(filename=os.path.join('..', 'output', self.example_name(), 'params.html'))

    def example_name(self):
        return ('example_' if self.mode == 'notebook' else '') + f'{self.dd.value}'

    def post_processing(self):
        with open(os.path.join('..', 'output', self.example_name(), 'params.json'), 'r') as f:
            params = json.load(f)

        opti_ceil = params['target_radius']
        # print(opti_ceil)

        factor = DEG_TO_RAD if params['coords'] == COORD_GCS else 1.
        target = factor * np.array(params['point_target'])
        # print(target)

        reach_time_nowind = params['geodesic_time']

        reach_time_pmp = float('inf')
        length_pmp = float('inf')
        reach_time_int = []
        length_int = []

        self.traj_stats = []

        with h5py.File(os.path.join('..', 'output', self.example_name(), 'trajectories.h5')) as f:
            for i, traj in enumerate(f.values()):
                coords = traj.attrs['coords']
                last_index = traj.attrs['last_index']
                reach_time = float('inf')
                notfound = True
                last_p = np.zeros(2)
                length = 0.
                for k, p in enumerate(traj['data']):
                    if k >= last_index:
                        break
                    p = factor * np.array(p)
                    if k > 0:
                        length += distance(p, last_p, coords=coords)
                    if notfound and distance(p, target, coords=coords) < 1.05 * opti_ceil:
                        reach_time = traj["ts"][k]
                        notfound = False
                    last_p[:] = p
                if traj.attrs['type'] in ['pmp', 'optimal']:
                    # If traj is an extremal, only look for the best one
                    reach_time_pmp = min(reach_time_pmp, reach_time)
                    length_pmp = min(length_pmp, length)
                else:
                    # If traj is a regular integral, add it straightforward
                    tst = {
                        'id': k,
                        'name': 'integral',
                        'duration': reach_time,
                        'length': length
                    }
                    self.traj_stats.append(tst)

        # Add best extremal
        tst = {
            'id': -1,
            'name': 'best extremal',
            'duration': reach_time_pmp,
            'length': length_pmp
        }
        self.traj_stats.append(tst)

        # Add the no-wind geodesic
        tst = {
            'id': -2,
            'name': 'no-wind geodesic',
            'duration': params['geodesic_time'],
            'length': 1.,
        }
        try:
            tst['length'] = params['geodesic_length']
        except KeyError:
            tst['length'] = reach_time_nowind / params['airspeed']
        self.traj_stats.append(tst)

        self.pp_params['reach_time_geodesic'] = reach_time_nowind
        self.pp_params['reach_time_pmp'] = reach_time_pmp
        self.pp_params['length_pmp'] = length_pmp
        for i, rti in enumerate(reach_time_int):
            self.pp_params[f'reach_time_int_{i}'] = rti
            self.pp_params[f'length_int_{i}'] = length_int[i]

    # def show_pp(self, units='hours'):
    #     reach_time_nowind = self.pp_params['reach_time_geodesic']
    #     reach_time_pmp = self.pp_params['reach_time_pmp']
    #     length_pmp = self.pp_params['length_pmp']
    #
    #     def ft(value):
    #         # value is expected in seconds
    #         if units == 'hours':
    #             return f'{value / 3600.:.2f} h'
    #         elif units == 'seconds':
    #             return f'{value:.2f} s'
    #         else:
    #             print(f'Unknown units "{units}"', file=sys.stderr)
    #             exit(1)
    #
    #     def fl(value):
    #         # value expected in meters
    #         if value > 1000.:
    #             return f'{value / 1000.:.2f} km'
    #         return f'{value:.2f} m'
    #
    #     print(f'   No wind geodesic : {ft(reach_time_nowind)}')
    #     print(f'     Reach time PMP : {ft(reach_time_pmp)}')
    #     print(f'         Length PMP : {fl(length_pmp)}')
    #     i = 0
    #     while True:
    #         try:
    #             reach_time_int = self.pp_params[f'reach_time_int_{i}']
    #             length_int = self.pp_params[f'length_int_{i}']
    #             print(f' Reach time int. {i:<2} : {ft(reach_time_int)}')
    #             print(f'     Length int. {i:<2} : {fl(length_int)}')
    #             print(f'     PMP saved time : {(reach_time_pmp - reach_time_int) / reach_time_int * 100:.1f} %')
    #             i += 1
    #         except KeyError:
    #             break

    def show_pp(self):
        def ft(value):
            # value is expected in seconds
            if value > 4000.:
                return f'{value / 3600.:.2f} h'
            return f'{value:.2f} s'

        def fl(value):
            # value expected in meters
            if value > 1000.:
                return f'{value / 1000.:.2f} km'
            return f'{value:.2f} m'

        s = "| Id | Name | Duration | Length | Mean groundspeed | \n|---|:---:|---|---|---|\n"
        for tst in self.traj_stats:
            s += f"| {tst['id']} | {tst['name']} | {ft(tst['duration'])} | {fl(tst['length'])} | {3.6 * tst['length'] / tst['duration']:.2f} km/h |\n"

        from IPython.core.display import display, Markdown, HTML
        # display(HTML("<style>.rendered_html { font-size: 30px; }</style>"))
        display(Markdown(s))


if __name__ == '__main__':
    fh = FrontendHandler(mode='default')
    fh.select_example(latest=False)
    fh.run_frontend()
    # fh.post_processing()
