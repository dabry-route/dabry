import argparse
import sys

import h5py
import numpy as np
import os


if __name__ == '__main__':
    steady = False
    parser = argparse.ArgumentParser()
    parser.add_argument('wind_fp')

    args = parser.parse_args(sys.argv[1:])

    with h5py.File(args.wind_fp, 'r') as f:
        bl = np.array((f['grid'][0, 0, 0], f['grid'][0, 0, 1]))
        tr = np.array((f['grid'][-1, -1, 0], f['grid'][-1, -1, 1]))
        values = np.array(f['data'])
        if steady and len(values.shape) == 4:
            values = values[0]
        if not steady:
            times = np.array(f['ts'])
        else:
            times = None

    dst_path = f'/home/bastien/Documents/work/flow/{os.path.basename(os.path.dirname(args.wind_fp))}'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    np.savez(os.path.join(dst_path, 'flow_unscaled'), values=values, bounds=np.stack((bl, tr)), times=times)
