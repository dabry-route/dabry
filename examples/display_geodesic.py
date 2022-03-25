from mdisplay.main import Display
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    output_path = '../output/example_geodesic'
    d = Display(coords='gcs', title='Geodesic approximation with extremals')
    d.setup(x_min=-76., x_max=5., y_min=30., y_max=60.)
    d.draw_trajs(os.path.join(output_path, 'trajectories.h5'))
    d.map.drawgreatcircle(-75., 40., 2., 48., linewidth=1, color='b', alpha=0.4,
                          linestyle='--',
                          label='Great circle', zorder=4)
    d.draw_point_by_name('New York')
    d.draw_point_by_name('Paris')
    plt.show()
