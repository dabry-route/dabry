import json
import time

import h5py
import os
import parse
import numpy as np
from datetime import datetime, timedelta

from src.wind import LinearWind


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
        print('{:<30}'.format('Getting data points...'), end='')
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
        print('Done')
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
                    self.data[k, i, j, 0] = wind_data['wind_u-surface'][0]
                    self.data[k, i, j, 1] = wind_data['wind_v-surface'][0]
                    self.ts[k] = wind_data['ts'][k]
        print('Done')

        self.grid = np.zeros((self.nx, self.ny, 2))
        for i in range(self.nx):
            for j in range(self.ny):
                self.grid[i, j, :] = np.array([self.lons[i], self.lats[j]])

    def dump(self, filepath):
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
            f.attrs.create('type', 'gcs')

            dset = f.create_dataset("data", (self.nt, self.nx, self.ny, 2), dtype='f8')
            dset[:, :, :, :] = self.data

            dset = f.create_dataset("ts", (self.nt,), dtype='i8')
            dset[:] = self.ts

            dset = f.create_dataset("grid", (self.nx, self.ny, 2), dtype='f8')
            dset[:, :, :] = self.grid


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


class WindHandler:
    """
    Handles wind loading from H5 format and derivative computation
    """

    def __init__(self):
        self.nt = None
        self.nx = None
        self.ny = None

        self.uv = None
        self.grid = None

        self.d_u__d_x = None
        self.d_u__d_y = None
        self.d_v__d_x = None
        self.d_v__d_y = None

    def load(self, filepath):
        """
        Loads wind data from H5 wind data
        :param filepath: The H5 file contaning wind data
        """

        # Fill the wind data array
        print(f'{"Loading wind values...":<30}', end='')
        with h5py.File(filepath, 'r') as wind_data:
            self.nt, self.nx, self.ny, _ = wind_data['data'].shape
            self.uv = np.zeros((self.nt, self.nx, self.ny, 2))
            self.uv[:, :, :, :] = wind_data['data']
            self.grid = np.zeros((self.nx, self.ny, 2))
            self.grid[:] = wind_data['grid']
        print('Done')

    def compute_derivatives(self):
        """
        Computes the derivatives of the windfield with a central difference scheme
        on the wind native grid
        """
        if self.uv is None:
            raise RuntimeError("Derivative computation failed : wind not yet loaded")
        self.d_u__d_x = np.zeros((self.nt, self.nx - 2, self.ny - 2))
        self.d_u__d_y = np.zeros((self.nt, self.nx - 2, self.ny - 2))
        self.d_v__d_x = np.zeros((self.nt, self.nx - 2, self.ny - 2))
        self.d_v__d_y = np.zeros((self.nt, self.nx - 2, self.ny - 2))

        # Use order 2 precision derivative
        self.d_u__d_x = 0.5 * (self.uv[:, 2:, 1:-1, 0] - self.uv[:, :-2, 1:-1, 0]) / (
                self.grid[2:, 1:-1, 0] - self.grid[:-2, 1:-1, 0])
        self.d_u__d_y = 0.5 * (self.uv[:, 1:-1, 2:, 0] - self.uv[:, 1:-1, :-2, 0]) / (
                self.grid[1:-1, 2:, 1] - self.grid[1:-1, :-2, 1])
        self.d_v__d_x = 0.5 * (self.uv[:, 2:, 1:-1, 1] - self.uv[:, :-2, 1:-1, 1]) / (
                self.grid[2:, 1:-1, 0] - self.grid[:-2, 1:-1, 0])
        self.d_v__d_y = 0.5 * (self.uv[:, 1:-1, 2:, 1] - self.uv[:, 1:-1, :-2, 1]) / (
                self.grid[1:-1, 2:, 1] - self.grid[1:-1, :-2, 1])


if __name__ == '__main__':
    """
    wc = WindConverter()
    wc.load('/home/bastien/Documents/data/wind/windy/Dakar-Natal-0.5')
    wc.dump('/home/bastien/Documents/data/wind/mermoz/Dakar-Natal-0.5')
    """
    lw = LinearWindExample()
    lw.dump('/home/bastien/Documents/data/wind/mermoz/linear-example')
