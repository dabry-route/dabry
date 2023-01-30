import json
import os
import shutil
import time

import markdown
import datetime

from dabry.problem import NavigationProblem
from dabry.solver import Solver
from dabry.solver_rp import SolverRP
from dabry.misc import Utils


class ParamsSummary:

    def __init__(self, params=None, style='params.css'):
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
        self.module_dir = None
        self.output_dir = None
        self.params_fname = 'params.json'
        self.params_ss_path = None
        self.params_fname_formatted = 'params.html'

        self.params = params if params is not None else {}
        self.md = None

    def setup(self, output_dir):
        path = os.environ.get('DABRYPATH')
        if path is None:
            raise Exception('Unable to set output dir automatically. Please set DABRYPATH variable.')
        self.module_dir = path
        self.params_ss_path = os.path.join(self.module_dir, 'docs')
        self.output_dir = output_dir

    def add_param(self, k, v):
        self.params[k] = v

    def load_dict(self, d):
        self.params = {}
        for k, v in d.items():
            self.params[k] = v

    def dump(self, fname=None, nohtml=False):
        # self.add_param('gen_time', time.time())
        self.process_params()
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

    def load_from_solver(self, sv: Solver):
        T = sv.T
        factor = Utils.RAD_TO_DEG if sv.mp.coords == Utils.COORD_GCS else 1.
        self.params = {
            'coords': sv.mp.coords,
            'point_init': (factor * sv.x_init[0], factor * sv.x_init[1]),
            'max_time': T,
            'airspeed': sv.mp.model.v_a,
            'point_target': (factor * sv.x_target[0], factor * sv.x_target[1]),
            'target_radius': sv.opti_ceil,
            'nt_pmp': sv.nt_pmp,
        }

    def load_from_solver_rp(self, sv: SolverRP):
        self.load_from_problem(sv.mp)
        self.add_param('max_time', sv.T)
        self.add_param('airspeed', sv.mp.model.v_a)
        self.add_param('target_radius', sv.opti_ceil)
        self.add_param('nt_pmp', sv.nt_pmp)
        self.add_param('nt_rft', sv.nt_rft)
        self.add_param('nx_rft', sv.nx_rft)
        self.add_param('ny_rft', sv.ny_rft)
        self.add_param('nt_rft_eff', sv.nt_rft_eff)

    def load_from_problem(self, pb: NavigationProblem):
        self.add_param('coords', pb.coords)
        self.add_param('x_init', tuple(pb.x_init))
        self.add_param('x_target', tuple(pb.x_target))
        self.add_param('airspeed', pb.model.v_a)
        self.add_param('target_radius', 0.05 * pb.geod_l)
        self.add_param('geodesic_time', pb.geod_l / pb.model.v_a)
        self.add_param('geodesic_length', pb.geod_l)
        self.add_param('aero_mode', pb.aero.mode)

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

        units_max_time = ''

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
            max_time = params['max_time']
            if max_time > 1800.:
                self.max_time = f'{max_time / 3600.:.2f}'
                units_max_time = 'h'
            else:
                self.max_time = max_time
                units_max_time = 's'
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
            f"| Time window upper bound | {self.max_time} | {units_max_time} |\n" + \
            f"| PMP number of time steps | {self.nt_pmp} |  |\n" + \
            f"| RFT number of time steps | {self.nt_rft} |  |\n" + \
            f"| RFT grid {self.coords_name} | {self.grid_rft} |  |\n" + \
            f"| Computation time PMP | {self.pmp_time} | s |\n" + \
            f"| Computation time RFT | {self.rft_time} | s |\n"

        s += table
        s += f'<link href="{self.params_ss_fname}" rel="stylesheet"/>'
        self.md = markdown.markdown(s, extensions=['tables'])
