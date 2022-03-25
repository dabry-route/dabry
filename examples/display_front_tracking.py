from mdisplay.main import Display
import os
import matplotlib.pyplot as plt
import h5py
from geopy import Nominatim
import markdown
import webbrowser

if __name__ == '__main__':
    output_path = '../output/example_front_tracking'
    loc = Nominatim(user_agent='openstreetmaps')
    d = Display(coords='gcs', title='Front tracking example')
    d.set_output_path(output_path)

    bl_name = 'Honolulu'
    bl_offset = (-5., -5.)
    tr_name = 'Vancouver'
    tr_offset = (5., 5.)

    d.setup(x_min=loc.geocode(bl_name).longitude + bl_offset[0],
            y_min=loc.geocode(bl_name).latitude + bl_offset[1],
            x_max=loc.geocode(tr_name).longitude + tr_offset[0],
            y_max=loc.geocode(tr_name).latitude + tr_offset[1])

    d.draw_wind('wind.h5')
    d.draw_rff('rff.h5')
    with h5py.File(os.path.join(output_path, 'rff.h5'), 'r') as f:
        nt_tick = f['ts'].shape[0]
        max_time = f['ts'][-1]
    print(max_time)
    d.draw_trajs('trajs.h5', nt_tick=nt_tick, max_time=max_time)

    d.draw_point_by_name(bl_name)
    d.draw_point_by_name(tr_name)

    d.show_params('params.json')

    plt.show()
