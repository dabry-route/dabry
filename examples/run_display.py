import sys
sys.path.append('/home/bastien/Documents/work/mdisplay/src')
from mdisplay.main import Display
import matplotlib.pyplot as plt

if __name__ == '__main__':
    d = Display(coords='gcs')
    d.setup(x_min=-76., x_max=5., y_min=35., y_max=52.)
    # d.draw_wind('/home/bastien/Documents/data/wind/mermoz/Vancouver-Honolulu-1.0/data.h5')
    d.draw_trajs('/home/bastien/Documents/work/mermoz/output/trajs/trajectories.h5')
    d.map.drawgreatcircle(-74., 40., 2., 48., linewidth=1, color='b', alpha=0.4,
                          linestyle='--',
                          label='Great circle')
    plt.show()