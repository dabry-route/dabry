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

    def __init__(self):
        self.display = None
        self.case_name = None
        self.dd = None
        self.output_path = None
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

        elif self.case_name == 'example_front_tracking':
            self.display.set_coords('gcs')
            self.display.set_title('Front tracking example')

            self.display.setup(projection='lcc')

            self.display.load_params()
            self.display.draw_wind()
            self.display.draw_rff()
            self.display.draw_trajs(nolabels=True)
            self.display.draw_point_by_name('Natal')

        elif self.case_name == 'example_front_tracking2' or self.case_name == 'example_front_tracking_linearinterp':
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
            self.display.draw_trajs(nolabels=False)
            self.display.draw_solver()
            lon_ny, lat_ny = self.display.geodata.get_coords('New York')
            lon_par, lat_par = self.display.geodata.get_coords('Paris')
            self.display.map.drawgreatcircle(lon_ny, lat_ny, lon_par, lat_par, linewidth=1, color='b', alpha=0.4,
                                             linestyle='--',
                                             label='Great circle', zorder=4)
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
        elif self.case_name in ['example_solver_double-gyre-kularatne2016', 'example_solver_double-gyre-li2020']:
            self.display.nocontrols = True
            self.display.set_title(f'{self.case_name.split("-")[-1]}')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind(autoscale=True, wind_nointerp=False)
            self.display.draw_trajs(nolabels=True, opti_only=opti_only)
            self.display.draw_rff()
            self.display.draw_solver()

            def on_plot_hover(event):
                # Iterating over each data member plotted
                for curve in self.display.mainax.get_lines():
                    # Searching which data member corresponds to current mouse position
                    if curve.contains(event)[0]:
                        with h5py.File(self.display.trajs_fpath, 'r') as f:
                            value = RAD_TO_DEG * f[str(curve.get_gid())]["controls"][0]
                            self.display.ax_info.set_text(str(value))
                            print(f'{value}')
                        break

            self.display.mainfig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

        elif self.case_name in ['example_solver_rad_gauss_1']:
            self.display.nocontrols = True
            self.display.set_title('Test case')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=False, opti_only=False)
            self.display.draw_solver()
            self.display.draw_rff()

        elif 'solver' in self.case_name:
            print(f'Using default solver setup script for unknown case "{self.case_name}"', file=sys.stderr)
            self.display.nocontrols = True
            self.display.set_title('Solver')
            self.display.load_params()
            self.display.setup(projection='ortho')
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=False, opti_only=True)
            self.display.draw_solver()
        else:
            print(f'Using default setup script for unknown case "{self.case_name}"', file=sys.stderr)
            self.display.nocontrols = True
            self.display.set_title('Example')
            self.display.load_params()
            self.display.setup()
            self.display.draw_wind()
            self.display.draw_trajs(nolabels=True)
            self.display.draw_rff()

    def select_example(self, mode='notebook'):
        if mode == 'notebook':
            from ipywidgets import Dropdown
            from IPython.core.display import display
            option_list = sorted(os.listdir(os.path.join('..', 'output')))
            new_ol = []
            for e in option_list:
                if not e.startswith('example_'):
                    del e
                else:
                    new_ol.append(e.split('example_')[1])
            self.dd = Dropdown(description="Choose one:", options=new_ol)
            self.dd.observe(lambda _: 0., names='value')
            display(self.dd)
        else:
            class ns:
                def __init__(self, value):
                    self.__dict__.update(value=value)

            self.dd = ns(
                os.path.basename(easygui.diropenbox(default=os.path.join('..', 'output'))).split('example_')[1])

    def run_frontend(self, mode='default', ex_name=None, noparams=True, opti_only=False):
        if mode == 'default':
            if ex_name is None:
                self.output_path = easygui.diropenbox(default=os.path.join('..', 'output'))
                if self.output_path is None:
                    exit(0)
            else:
                self.output_path = os.path.join('..', 'output', ex_name)
            self.case_name = os.path.basename(self.output_path)
        elif mode == 'notebook':
            self.output_path = os.path.join('..', 'output', f'example_{self.dd.value}')
            self.case_name = os.path.basename(self.output_path)
        else:
            print(f'Unknown mode {mode}', file=sys.stderr)
            exit(1)

        self.display = Display()
        self.display.set_output_path(self.output_path)
        self.configure(opti_only=opti_only)
        self.display.update_title()
        self.display.show(noparams=noparams)

    def show_params(self):
        from IPython.core.display import HTML
        return HTML(filename=os.path.join('..', 'output', 'example_' + self.dd.value, 'params.html'))

    def post_processing(self):
        with open(os.path.join('..', 'output', f'example_{self.dd.value}', 'params.json'), 'r') as f:
            params = json.load(f)

        opti_ceil = params['target_radius']

        factor = DEG_TO_RAD if params['coords'] == COORD_GCS else 1.
        target = factor * np.array(params['point_target'])

        reach_time_nowind = params['geodesic_time']
        reach_time_pmp = float('inf')
        reach_time_int = []

        with h5py.File(os.path.join('..', 'output', f'example_{self.dd.value}', 'trajectories.h5')) as f:
            for i, traj in enumerate(f.values()):
                reach_time = float('inf')
                for k, p in enumerate(traj['data']):
                    p = factor * np.array(p)
                    if params['coords'] == COORD_GCS and geodesic_distance(p, target, mode='rad') < opti_ceil \
                            or np.linalg.norm(p - target) < opti_ceil:
                        reach_time = traj["ts"][k]
                        break
                if traj.attrs['type'] in ['pmp', 'optimal']:
                    reach_time_pmp = min(reach_time_pmp, reach_time)
                else:
                    reach_time_int.append(reach_time)

        self.pp_params['reach_time_geodesic'] = reach_time_nowind
        self.pp_params['reach_time_pmp'] = reach_time_pmp
        for i, rti in enumerate(reach_time_int):
            self.pp_params[f'reach_time_int_{i}'] = rti

    def show_pp(self, units='hours'):
        reach_time_nowind = self.pp_params['reach_time_geodesic']
        reach_time_pmp = self.pp_params['reach_time_pmp']

        def ft(value):
            # value is expected in seconds
            if units == 'hours':
                return f'{value / 3600.:.2f} h'
            elif units == 'seconds':
                return f'{value:.2f} s'
            else:
                print(f'Unknown units "{units}"', file=sys.stderr)
                exit(1)

        print(f'   No wind geodesic : {ft(reach_time_nowind)}')
        print(f'     Reach time PMP : {ft(reach_time_pmp)}')
        i = 0
        while True:
            try:
                reach_time_int = self.pp_params[f'reach_time_int_{i}']
                print(f' Reach time int. {i:<2} : {ft(reach_time_int)}')
                print(f'     PMP saved time : {(reach_time_pmp - reach_time_int) / reach_time_int * 100:.1f} %')
                i += 1
            except KeyError:
                break


if __name__ == '__main__':
    fh = FrontendHandler()
    fh.select_example(mode='runtime')
    fh.run_frontend()
    fh.post_processing()
