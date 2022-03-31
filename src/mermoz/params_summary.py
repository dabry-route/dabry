import json
import os
import shutil

import markdown
import datetime


class ParamsSummary:

    def __init__(self, params, output_dir, style='params.css'):
        self.coords_units = None
        self.x_name = None
        self.y_name = None

        self.coords = None
        self.coords_name = None
        self.start_point = None
        self.bl_wind = None
        self.tr_wind = None
        self.grid_wind = None
        self.date_wind = None
        self.max_time = None
        self.nt_pmp = None
        self.nt_rft = None
        self.grid_rft = None
        self.pmp_time = None
        self.rft_time = None

        self.params_ss_fname = style
        self.output_dir = output_dir
        self.params_fname = 'params.json'
        self.params_ss_path = '/home/bastien/Documents/work/mermoz/docs'
        self.params_fname_formatted = 'params.html'

        self.params = params
        self.md = None
        self.process_params()

    def dump(self, fname=None, nohtml=False):
        fname = self.params_fname if fname is None else fname
        with open(os.path.join(self.output_dir, fname), 'w') as f:
            json.dump(self.params, f)
        if not nohtml:
            path = os.path.join(self.output_dir, self.params_fname_formatted)
            with open(path, "w", encoding="utf-8", errors="xmlcharrefreplace") as output_file:
                output_file.write(self.md)
            shutil.copyfile(os.path.join(self.params_ss_path, self.params_ss_fname),
                            os.path.join(self.output_dir, self.params_ss_fname))


    def _fcoords(self, coords):
        if type(coords) == str:
            return coords
        else:
            return f'{coords[0]:.3f}, {coords[1]:.3f}'

    def _fcomptime(self, comptime):
        return f'{comptime:.3f}'

    def process_params(self):
        params = self.params
        if params['coords'] == 'gcs':
            self.coords_units = '°'
            self.coords_name = '(lon, lat)'
        else:
            self.coords_units = 'm'
            self.coords_name = '(X, Y)'

        s = ""
        s += '### Computation parameters\n\n'

        try:
            self.coords = self._fcoords(params['coords'])
        except KeyError:
            pass
        try:
            self.start_point = self._fcoords(params['point_init'])
        except KeyError:
            pass
        try:
            self.bl_wind = self._fcoords(params['bl_wind'])
        except KeyError:
            pass
        try:
            self.tr_wind = self._fcoords(params['tr_wind'])
        except KeyError:
            pass
        try:
            self.grid_wind = f'{params["nx_wind"]}x{params["ny_wind"]}'
        except KeyError:
            pass
        try:
            self.date_wind = datetime.datetime.fromtimestamp(params['date_wind'] / 1000.)
        except KeyError:
            pass
        try:
            self.max_time = params['max_time']
        except KeyError:
            pass
        try:
            self.nt_pmp = params['nt_pmp']
        except KeyError:
            pass
        try:
            self.nt_rft = params['nt_rft']
        except KeyError:
            pass
        try:
            self.grid_rft = f"{params['nx_rft']}x{params['ny_rft']}"
        except KeyError:
            pass
        try:
            self.pmp_time = self._fcomptime(params['pmp_time'])
        except KeyError:
            pass
        try:
            self.rft_time = self._fcomptime(params['rft_time'])
        except KeyError:
            pass

        table = \
            "| Parameter         | Value | Units |\n|---|:---:|---|\n" + \
            f"| Type of coordinates  | {self.coords} |   |\n" + \
            f"| Start point | {self.start_point} | {self.coords_units} |\n" + \
            f"| Wind bottom left bound | {self.bl_wind} | {self.coords_units} |\n" + \
            f"| Wind top right bound | {self.tr_wind} | {self.coords_units} |\n" + \
            f"| Wind grid {self.coords_name} | {self.grid_wind} |  |\n" + \
            f"| Wind date | {self.date_wind} |  |\n" + \
            f"| Time window upper bound | {self.max_time} | s |\n" + \
            f"| PMP number of time steps | {self.nt_pmp} |  |\n" + \
            f"| RFT number of time steps | {self.nt_rft} |  |\n" + \
            f"| RFT grid {self.coords_name} | {self.grid_rft} |  |\n" + \
            f"| Computation time PMP | {self.pmp_time} | s |\n" + \
            f"| Computation time RFT | {self.rft_time} | s |\n"

        s += table
        s += f'<link href="{self.params_ss_fname}" rel="stylesheet"/>'
        self.md = markdown.markdown(s, extensions=['tables'])
