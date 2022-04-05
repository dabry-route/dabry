import os.path
import sys
import numpy as np
import easygui

sys.path.extend(['/home/bastien/Documents/work/mdisplay', '/home/bastien/Documents/work/mdisplay/src'])
from mdisplay.display import Display


def configure(d: Display, case_name: str):
    if case_name == 'example_geodesic':
        d.set_title('Geodesic approximation with extremals')
        d.set_coords('gcs')
        bl = np.array([-76., 30.])
        tr = np.array([5., 60.])
        d.setup(bl=bl, tr=tr)
        d.load_params()
        d.draw_trajs(nolabels=True)
        lon_ny, lat_ny = d.geodata.get_coords('New York')
        lon_par, lat_par = d.geodata.get_coords('Paris')
        d.map.drawgreatcircle(lon_ny, lat_ny, lon_par, lat_par, linewidth=1, color='b', alpha=0.4,
                              linestyle='--',
                              label='Great circle', zorder=4)
        d.draw_point_by_name('New York')
        d.draw_point_by_name('Paris')

    elif case_name == 'example_solver_linearwind' or case_name == 'example_ft_linearwind':
        d.set_coords('cartesian')
        d.set_title('Analytic solution comparison')
        d.setup()
        d.load_params()
        d.draw_wind()
        d.draw_trajs()
        d.draw_point(1e6, 0., label='Target')
        if case_name == 'example_ft_linearwind':
            d.draw_rff()

    elif case_name == 'example_front_tracking':
        d.set_coords('gcs')
        d.set_title('Front tracking example')

        d.setup(projection='lcc')

        d.load_params()
        d.draw_wind()
        d.draw_rff()
        d.draw_trajs(nolabels=True)
        d.draw_point_by_name('Dakar')
        d.draw_point_by_name('Natal')

    elif case_name == 'example_front_tracking2' or case_name == 'example_front_tracking_linearinterp':
        d.set_coords('gcs')
        d.set_title('Front tracking example')

        d.setup(projection='lcc')

        d.load_params()
        d.draw_wind()
        d.draw_rff()
        d.draw_trajs(nolabels=True)

        d.draw_point_by_name('Tripoli')
        d.draw_point_by_name('Athenes')

    elif case_name == 'example_front_tracking3':
        d.set_coords('gcs')
        d.set_title('Front tracking example')

        d.setup(projection='lcc')

        d.load_params()
        d.draw_wind()
        d.draw_rff()
        d.draw_trajs(nolabels=True)

        d.draw_point_by_name('New York')
        d.draw_point_by_name('Paris')

    elif case_name == 'example_front_tracking4':
        d.set_coords('gcs')
        d.set_title('Front tracking example')

        d.setup()

        d.load_params()
        d.draw_wind()
        d.draw_rff()
        d.draw_trajs(nolabels=True)

    elif case_name == 'example_solver':
        d.nocontrols = True
        d.set_coords('cartesian')
        d.set_title('Solver test')
        # bl = np.array([0.2, -0.1])
        # tr = np.array([1.2, 0.2])
        # d.setup(bl=bl, tr=tr)
        d.setup()
        d.load_params()
        d.draw_wind()
        d.draw_trajs(nolabels=True)
        d.draw_point(1e6, 0., label='Target')
        # d.legend()

    elif case_name == 'example_ft_vor3':
        d.nocontrols = True
        d.set_coords('cartesian')
        d.set_title('Front tracking on Rankine vortices')
        d.setup()
        d.load_params()
        d.draw_wind()
        d.draw_trajs(nolabels=True, filename='../example_solver/trajectories.h5')
        d.draw_rff()

    else:
        print(f'Using default setup script for unknown case "{case_name}"', file=sys.stderr)
        d.nocontrols = True
        d.set_coords('cartesian')
        d.set_title('Example')
        d.setup()
        d.load_params()
        d.draw_wind()
        d.draw_trajs(nolabels=True)
        d.draw_rff()


def select_example():
    from ipywidgets import Dropdown
    from IPython.core.display import display
    option_list = sorted(os.listdir(os.path.join('..', 'output')))
    new_ol = []
    for e in option_list:
        if not e.startswith('example_'):
            del e
        else:
            new_ol.append(e.split('example_')[1])
    dropdown = Dropdown(description="Choose one:", options=new_ol)
    dropdown.observe(lambda _: 0., names='value')
    display(dropdown)
    return dropdown


def run_frontend(mode='default', ex_name=None, noparams=False):
    if mode == 'default':
        if ex_name is None:
            output_path = easygui.diropenbox(default=os.path.join('..', 'output'))
            if output_path is None:
                exit(0)
        else:
            output_path = os.path.join('..', 'output', ex_name)
        bn = os.path.basename(output_path)
        d = Display()
        d.set_output_path(output_path)
        configure(d, bn)
        d.show(noparams=noparams)
    elif mode == 'notebook':
        output_path = os.path.join('..', 'output', ex_name)
        bn = os.path.basename(output_path)
        d = Display()
        d.set_output_path(output_path)
        configure(d, bn)
        d.update_title()
        d.show(noparams=True)
    else:
        print(f'Unknown mode {mode}', file=sys.stderr)
        exit(1)


if __name__ == '__main__':
    run_frontend()
