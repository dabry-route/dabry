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
        d.draw_trajs(nolabels=True)
        lon_ny, lat_ny = d.geodata.get_coords('New York')
        lon_par, lat_par = d.geodata.get_coords('Paris')
        d.map.drawgreatcircle(lon_ny, lat_ny, lon_par, lat_par, linewidth=1, color='b', alpha=0.4,
                              linestyle='--',
                              label='Great circle', zorder=4)
        d.draw_point_by_name('New York')
        d.draw_point_by_name('Paris')

    elif case_name == 'example_linearwind':
        d.set_coords('cartesian')
        d.set_title('Analytic solution comparison')
        bl = np.array([0.2, -0.1])
        tr = np.array([1.2, 0.2])
        d.setup(bl=bl, tr=tr)
        d.draw_wind()
        d.draw_trajs()
        d.draw_point(1e6, 0., label='Target')

    elif case_name == 'example_front_tracking':
        d.set_coords('gcs')
        d.set_title('Front tracking example')

        d.setup()

        d.load_params()
        d.draw_wind()
        d.draw_rff()
        d.draw_trajs(nolabels=True)
        d.draw_point_by_name('Dakar')
        d.draw_point_by_name('Natal')

    elif case_name == 'example_front_tracking2':
        d.set_coords('gcs')
        d.set_title('Front tracking example')

        d.setup()

        d.load_params()
        d.draw_wind()
        d.draw_rff()
        d.draw_trajs(nolabels=True)

        d.draw_point_by_name('Tripoli')
        d.draw_point_by_name('Athenes')

    elif case_name == 'example_solver':
        d.set_coords('cartesian')
        d.set_title('Solver test')
        bl = np.array([0.2, -0.1])
        tr = np.array([1.2, 0.2])
        d.setup(bl=bl, tr=tr)
        d.draw_wind()
        d.draw_trajs(nolabels=True)
        d.draw_point(1e6, 0., label='Target')

    else:
        raise FileNotFoundError(f'No setup script provided for case "{case_name}"')


if __name__ == '__main__':
    output_path = easygui.diropenbox(default='../output')
    if output_path is None:
        exit(0)
    bn = os.path.basename(output_path)
    d = Display()
    d.set_output_path(output_path)
    configure(d, bn)
    d.show()
