import json
import time
import argparse
import h5py
import os
import parse
import numpy as np
from datetime import datetime, timedelta
import shutil

from mermoz.misc import EARTH_RADIUS, COORDS, ensure_coords, COORD_CARTESIAN, U_METERS, COORD_GCS, U_RAD, U_DEG, \
    DEG_TO_RAD, ensure_units, ensure_compatible
from mermoz.wind import LinearWind


class WindConverter:
    """
    Handles conversion from Windy JSON wind format to
    Mermoz-specific H5 wind format (see docs/)
    """

    def __init__(self):
        self.lons = None
        self.lats = None
        self.nx = None
        self.ny = None
        self.nt = None
        self.data = None
        self.ts = None
        self.grid = None

    def load(self, path_to_wind):
        """
        Loads wind data from all files with name properly formatted in given dir.
        :param path_to_wind: The path to wind files
        """

        self.lons = []
        self.lats = []
        # Parse files to get all longitudes and latitudes
        print('{:<30}'.format('Listing data points...'), end='')
        for file in os.listdir(path_to_wind):
            expr = r'wind_{}_{}.txt'
            parsed = parse.parse(expr, file)
            lat, lon = list(map(lambda x: float(x), parsed))
            if lon not in self.lons:
                self.lons.append(lon)
            if lat not in self.lats:
                self.lats.append(lat)
        self.lons = sorted(self.lons)
        self.lats = sorted(self.lats)
        self.nx = len(self.lons)
        self.ny = len(self.lats)
        print(f'Done ({self.nx}x{self.ny}, total {self.nx * self.ny})')
        # Fill the wind data array
        print(f'{"Loading wind values...":<30}', end='')
        # Getting number of timestamps
        with open(os.path.join(path_to_wind, f'wind_{self.lats[0]}_{self.lons[0]}.txt'), 'r') as windfile:
            wind_data = json.load(windfile)
            self.nt = len(wind_data['ts'])

        # Filling wind values
        self.data = np.zeros((self.nt, self.nx, self.ny, 2))
        self.ts = np.zeros((self.nt,))
        for k in range(self.nt):
            for i, lon in enumerate(self.lons):
                for j, lat in enumerate(self.lats):
                    file = f'wind_{lat}_{lon}.txt'
                    with open(os.path.join(path_to_wind, file), 'r') as windfile:
                        wind_data = json.load(windfile)
                    try:
                        self.data[k, i, j, 0] = wind_data['wind_u-surface'][0]
                        self.data[k, i, j, 1] = wind_data['wind_v-surface'][0]
                    except KeyError:
                        self.data[k, i, j, 0] = wind_data['wind_u-1000h'][0]
                        self.data[k, i, j, 1] = wind_data['wind_v-1000h'][0]
                    self.ts[k] = wind_data['ts'][k]
        print('Done')

        self.grid = np.zeros((self.nx, self.ny, 2))
        for i in range(self.nx):
            for j in range(self.ny):
                self.grid[i, j, :] = np.array([self.lons[i], self.lats[j]])

    def convert(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Source not found : {filepath}')
        self.load(filepath)
        dst = os.path.realpath(filepath) + '.mz'
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.mkdir(dst)
        self.dump(dst)

    def dump(self, filepath):

        print(f'{f"Dumping to {filepath}...":<30}', end='')
        config = ""
        config += '### Wind data collection\n'
        config += f'Generated on {datetime.fromtimestamp(time.time()).ctime()}\n\n'
        table = \
            "| Parameter         | Value |\n|---|---|\n" + \
            f"| First timestamp   | {datetime.fromtimestamp(self.ts[0] / 1000)} |\n" + \
            f"| Time window width | {timedelta(seconds=(self.ts[-1] - self.ts[0]) / 1000)} |\n"
        config += table

        with open(os.path.join(filepath, 'config.md'), 'w') as f:
            f.writelines(config)

        with h5py.File(os.path.join(filepath, 'data.h5'), "w") as f:
            f.attrs.create('coords', 'gcs')
            f.attrs.create('units_grid', 'degrees')

            dset = f.create_dataset("data", (self.nt, self.nx, self.ny, 2), dtype='f8')
            dset[:, :, :, :] = self.data

            dset = f.create_dataset("ts", (self.nt,), dtype='i8')
            dset[:] = self.ts

            dset = f.create_dataset("grid", (self.nx, self.ny, 2), dtype='f8')
            dset[:, :, :] = self.grid

        print('Done')


class LinearWindExample:
    def __init__(self):
        self.gradient = np.array([[0., 23. / 10.],
                                  [0., 0.]])
        self.origin = np.array([0., 0.])
        self.value_origin = np.array([0., 0.])
        self.nt = 1
        self.nx = 100
        self.ny = 100
        self.x_min = 0.
        self.x_max = 1.
        self.y_min = 0.
        self.y_max = 1.
        lwind = LinearWind(self.gradient, self.origin, self.value_origin)
        self.data = np.zeros((self.nt, self.nx, self.ny, 2))
        self.grid = np.zeros((self.nx, self.ny, 2))
        self.ts = np.zeros((1,))
        for i, x in enumerate(np.linspace(self.x_min, self.x_max, self.nx)):
            for j, y in enumerate(np.linspace(self.y_min, self.y_max, self.ny)):
                self.data[0, i, j, :] = lwind.value(np.array([x, y]))
                self.grid[i, j, :] = np.array([x, y])

    def dump(self, filepath):
        with h5py.File(os.path.join(filepath, 'wind.h5'), "w") as f:
            dset = f.create_dataset("data", (self.nt, self.nx, self.ny, 2), dtype='f8')
            dset[:, :, :, :] = self.data

            dset = f.create_dataset("ts", (self.nt,), dtype='i8')
            dset[:] = self.ts

            dset = f.create_dataset("grid", (self.nx, self.ny, 2), dtype='f8')
            dset[:, :, :] = self.grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str)
    args = parser.parse_args()
    wc = WindConverter()
    wc.convert(args.src)
