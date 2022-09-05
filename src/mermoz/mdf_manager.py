import datetime
import os
import pygrib

import h5py

from mermoz.misc import *
from mermoz.wind import Wind, DiscreteWind


class MDFmanager:
    """
    This class handles the writing and reading of Mermoz Data Format (MDF) files
    """

    def __init__(self):
        self.output_dir = None
        self.trajs_filename = 'trajectories.h5'
        self.wind_filename = 'wind.h5'

    def set_output_dir(self, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir

    def clean_output_dir(self, keep_rff=False):
        for filename in os.listdir(self.output_dir):
            if not keep_rff or not filename.endswith('rff.h5'):
                os.remove(os.path.join(self.output_dir, filename))

    def dump_trajs(self, traj_list, filename=None):
        filename = self.trajs_filename if filename is None else filename
        filepath = os.path.join(self.output_dir, filename)
        with h5py.File(filepath, "a") as f:
            index = 0
            if len(f.keys()) != 0:
                index = int(max(f.keys(), key=int)) + 1
            for i, traj in enumerate(traj_list):
                nt = traj.timestamps.shape[0]
                trajgroup = f.create_group(str(index + i))
                trajgroup.attrs['type'] = traj.type
                trajgroup.attrs['coords'] = traj.coords
                trajgroup.attrs['interrupted'] = traj.interrupted
                trajgroup.attrs['last_index'] = traj.last_index
                trajgroup.attrs['label'] = traj.label
                trajgroup.attrs['info'] = traj.info

                dset = trajgroup.create_dataset('data', (nt, 2), dtype='f8')
                dset[:, :] = traj.points

                dset = trajgroup.create_dataset('ts', (nt,), dtype='f8')
                dset[:] = traj.timestamps

                dset = trajgroup.create_dataset('controls', (nt,), dtype='f8')
                dset[:] = traj.controls

                if hasattr(traj, 'adjoints'):
                    dset = trajgroup.create_dataset('adjoints', (nt, 2), dtype='f8')
                    dset[:, :] = traj.adjoints

    def _grib_date_to_unix(self, grib_filename):
        date, hm = grib_filename.split('_')[2:4]
        hours = int(hm[:2])
        minutes = int(hm[2:])
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:])
        dt = datetime.datetime(year, month, day, hours, minutes)
        return dt.timestamp()

    def print_trajs(self, filepath):
        with h5py.File(filepath, 'r') as f:
            for traj in f.values():
                print(traj)
                for attr, val in traj.attrs.items():
                    print(f'{attr} : {val}')

    def dump_wind(self, wind: Wind, filename=None, nx=None, ny=None, nt=None, bl=None, tr=None, coords=COORD_CARTESIAN, force_analytical=False):
        if wind.is_dumpable == 0:
            print('Error : Wind is not dumpable to file', file=sys.stderr)
            exit(1)
        filename = self.wind_filename if filename is None else filename
        filepath = os.path.join(self.output_dir, filename)
        if wind.is_dumpable == 1:
            if nx is None or ny is None:
                print(f'Please provide grid shape "nx=..., ny=..." to sample analytical wind "{wind}"')
                exit(1)
            dwind = DiscreteWind(force_analytical=force_analytical)
            kwargs = {}
            if nt is not None:
                kwargs['nt'] = nt
            dwind.load_from_wind(wind, nx, ny, bl, tr, coords, nodiff=True, **kwargs)
        else:
            dwind = wind
        with h5py.File(filepath, 'w') as f:
            f.attrs['coords'] = dwind.coords
            f.attrs['units_grid'] = U_METERS if dwind.coords == COORD_CARTESIAN else U_RAD
            f.attrs['analytical'] = dwind.is_analytical
            if dwind.lon_0 is not None:
                f.attrs['lon_0'] = dwind.lon_0
            if dwind.lat_0 is not None:
                f.attrs['lat_0'] = dwind.lat_0
            f.attrs['unstructured'] = dwind.unstructured
            dset = f.create_dataset('data', (dwind.nt, dwind.nx, dwind.ny, 2), dtype='f8')
            dset[:] = dwind.uv
            dset = f.create_dataset('ts', (dwind.nt,), dtype='f8')
            dset[:] = dwind.ts
            dset = f.create_dataset('grid', (dwind.nx, dwind.ny, 2), dtype='f8')
            dset[:] = dwind.grid

    def dump_wind_from_grib2(self, srcfiles, bl, tr, dstname=None, coords=COORD_GCS):
        if coords == COORD_CARTESIAN:
            print('Cartesian conversion not handled yet', file=sys.stderr)
            exit(1)
        if type(srcfiles) == str:
            srcfiles = [srcfiles]
        filename = self.wind_filename if dstname is None else dstname
        filepath = os.path.join(self.output_dir, filename)

        def process(grbfile, setup=False, nx=None, ny=None):
            grbs = pygrib.open(grbfile)
            grb = grbs.select(name='U component of wind', typeOfLevel='isobaricInhPa', level=1000)[0]
            lon_b = rectify(bl[0], tr[0])
            U, lats, lons = grb.data(lat1=bl[1], lat2=tr[1], lon1=lon_b[0], lon2=lon_b[1])
            grb = grbs.select(name='V component of wind', typeOfLevel='isobaricInhPa', level=1000)[0]
            V, _, _ = grb.data(lat1=bl[1], lat2=tr[1], lon1=lon_b[0], lon2=lon_b[1])
            if setup:
                if lons.max() > 180.:
                    newlons = np.zeros(lons.shape)
                    newlons[:] = lons - 360.
                    lons[:] = newlons

                newlats = np.zeros(lats.shape)
                newlats[:] = lats[::-1, :]
                lats[:] = newlats

                ny, nx = U.shape
                grid = np.zeros((nx, ny, 2))
                grid[:] = np.transpose(np.stack((lons, lats)), (2, 1, 0))
                return nx, ny, grid
            else:
                if nx is None or ny is None:
                    print('Missing nx or ny', file=sys.stderr)
                    exit(1)
                UV = np.zeros((nx, ny, 2))
                UV[:] = np.transpose(np.stack((U, V)), (2, 1, 0))
                return UV

        # First fetch grid parameters
        nx, ny, grid = process(srcfiles[0], setup=True)
        nt = len(srcfiles)
        UVs = np.zeros((nt, nx, ny, 2))
        dates = np.zeros((nt,))
        for k, grbfile in enumerate(srcfiles):
            UV = process(grbfile, nx=nx, ny=ny)
            UVs[k, :] = UV
            dates[k] = self._grib_date_to_unix(os.path.basename(grbfile))

        with h5py.File(filepath, 'w') as f:
            f.attrs['coords'] = coords
            f.attrs['units_grid'] = U_RAD
            f.attrs['analytical'] = False
            dset = f.create_dataset('data', (nt, nx, ny, 2), dtype='f8')
            dset[:] = UVs
            dset = f.create_dataset('ts', (nt,), dtype='f8')
            dset[:] = dates
            dset = f.create_dataset('grid', (nx, ny, 2), dtype='f8')
            dset[:] = DEG_TO_RAD * grid

        return nt, nx, ny


if __name__ == '__main__':
    mdfm = MDFmanager()
    # mdfm.print_trajs('/home/bastien/Documents/work/mermoz/output/example_front_tracking2/trajectories.h5')
    mdfm.set_output_dir('/home/bastien/Documents/work/test')
    data_dir = '/home/bastien/Documents/data/other'
    # data_filepath = os.path.join(data_dir, 'gfs_3_20090823_0600_000.grb2')
    data_filepath = os.path.join(data_dir, 'gfs_4_20220324_1200_000.grb2')
    bl = np.array((-50., -40.))
    tr = np.array((10., 40.))
    mdfm.dump_wind_from_grib2(data_filepath, bl, tr)
