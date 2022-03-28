from mdisplay.display import Display
import os
import matplotlib.pyplot as plt
import h5py
import markdown
import webbrowser

if __name__ == '__main__':
    output_path = '../output/example_front_tracking2'
    d = Display(coords='gcs', title='Front tracking example')
    d.set_output_path(output_path)

    d.setup()

    d.draw_wind('wind.h5')
    d.draw_rff('rff.h5')
    with h5py.File(os.path.join(output_path, 'rff.h5'), 'r') as f:
        nt_tick = f['ts'].shape[0]
        max_time = f['ts'][-1]
    print(max_time)
    d.draw_trajs('trajs.h5', nt_tick=nt_tick, max_time=max_time)

    d.draw_point_by_name('Dakar')
    d.draw_point_by_name('Natal')

    d.show_params()

    plt.show()
