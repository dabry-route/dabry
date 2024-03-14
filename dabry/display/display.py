import os
import sys
import time
import warnings
from datetime import datetime
from enum import Enum
from math import floor
from typing import Optional, Dict

import h5py
import matplotlib
import matplotlib.cm as mpl_cm
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from mpl_toolkits.basemap import Basemap
from pyproj import Proj, Geod
from scipy.interpolate import griddata

from dabry.display.misc import *
from dabry.flowfield import DiscreteFF
from dabry.io_manager import IOManager
from dabry.misc import Utils, Coords
from dabry.obstacle import DiscreteObs
from dabry.trajectory import Trajectory, traj_filepath_to_name


class ZOrder(Enum):
    WIND_NORM = 1
    WIND_VECTORS = 2
    RFF = 3
    WIND_ANCHORS = 4
    OBS = 5
    TRAJS = 6
    ANNOT = 7


class FontsizeConf:

    def __init__(self):
        self.fontsize = 18
        self.axes_titlesize = 24
        self.axes_labelsize = 22
        self.xtick_labelsize = 22
        self.ytick_labelsize = 22
        self.legend_fontsize = 15
        self.button_fontsize = 14
        self.timedisp_major = 20
        self.timedisp_minor = 14
        self.font_family = 'lato'
        self.mathtext_fontset = 'cm'


class Display:
    """
    Handles the visualization of navigation problems
    """

    def __init__(self, name: str, case_dir: Optional[str] = None, mode_3d=False):
        self.io = IOManager(name, case_dir=case_dir)

        self.x_offset = 0.
        self.y_offset = 0.

        self.x_offset_gcs = 0.2
        self.y_offset_gcs = 0.2

        self.main_fig: Optional[Figure] = None
        self.main_ax: Optional[Axes] = None

        self.fsc = None

        # The object that handles plot, scatter, contourf... whatever cartesian or gcs
        self.ax: Optional[Axes] = None

        # Adjoint figure
        self.map: Optional[Basemap] = None
        self.display_setup: bool = False
        self.cm = None
        self.selected_cm = windy_cm  # custom_cm  # windy_cm
        self.cm_norm_min = 0.
        self.cm_norm_max = 24  # 46.
        self.cm_norm_min_mag = 5
        self.airspeed = None
        self.title = ''
        self.axes_equal = True
        self.projection = 'merc'
        self.sm_ff = None
        self.sm_engy = None
        self.leg = None
        self.leg_handles = []
        self.leg_labels = []

        self.bl_man = None
        self.tr_man = None

        # Number of expected time major ticks on trajectories
        self.nt_tick = None
        self.t_tick = None

        self.p_names = ['x_init', 'x_target', 'target_radius']
        # Dictionary reading case parameters
        self.params = {}

        self.case_dir = case_dir
        self.img_params = {
            'dpi': 300,
            'format': 'png'
        }

        self.ff_fname = 'ff.h5'
        self.rff_fname = 'rff.h5'
        self.obs_fname = 'obs.h5'
        self.pen_fname = 'penalty.h5'
        self.filter_fname = '.trajfilter'
        self.ff_fpath = None
        self.trajs_fpath = None
        self.rff_fpath = None
        self.obs_fpath = None
        self.pen_fpath = None
        self.filter_fpath = None

        self.traj_filter = []
        # 0 for no aggregation (fronts), 1 for aggregation and time cursor, 2 for aggrgation and no time cursor
        self.mode_aggregated = False
        self.mode_controls = False  # Whether to display controls
        self.mode_ef = True  # Whether to display zones where ff is equal to airspeed
        self.mode_annot = False  # Whether to display trajectories annotation
        self.mode_ff_color = True  # Whether to display ff colors
        self.mode_energy = False  # Whether to display energy as colors
        self.mode_ef_display = True  # Whether to draw extremal fields or not
        self.rescale_ff = True  # Whether to rescale ff

        # True if ff norm colobar is displayed, False if energy colorbar is displayed
        self.active_ffcb = True

        self.has_display_rff = True

        self.ff = None
        self.ff_norm_min = None
        self.ff_norm_avg = None
        self.ff_norm_max = None
        self.trajs: Dict[str, Trajectory] = {}
        self.trajs_regular: Dict[str, Trajectory] = {}
        self.rff = None
        self.rff_cntr_kwargs = {
            'zorder': ZOrder.RFF.value,
            'cmap': 'brg',
            'antialiased': True,
            'alpha': .8,
        }
        self.rff_zero_ceils = None
        self.nx_rft = None
        self.ny_rft = None
        self.nt_rft = None

        self.obstacles = None
        self.obs_total_values = None
        self.obs_grid = None

        self.penalty = None

        self.traj_artists = []
        self.traj_lines = []
        self.traj_ticks = []
        self.traj_lp = []
        self.traj_annot = []
        self.traj_controls = []
        self.traj_epoints = []
        self.id_traj_color = 0

        self.rff_contours = []
        self.obs_contours = []
        self.pen_contours = []

        self.scatter_init = None
        self.scatter_target = None
        self.circle_target = None

        self.ax_rbutton = None
        self.reload_button = None
        self.ax_sbutton = None
        self.switch_button = None
        self.ax_cbutton = None
        self.control_button = None
        self.ax_info = None

        self.has_manual_bbox = False

        self.engy_min = self.engy_max = None
        self.engy_norm = None

        self.ff_colormesh = None
        self.ff_colorcontour = None
        self.ff_ceil = None
        self.ff_quiver = None
        self.ax_timeslider = None
        self.time_slider = None
        self.ff_colorbar = None
        self.ax_timedisplay = None
        self.ax_timedisp_minor = None
        # Time window lower bound
        # Defined as ff time lower bound if ff is time-varying
        # Else minimum timestamps among trajectories and fronts
        self.tl = None
        # Time window upper bound
        # Maximum among all upper bounds (ff, trajs, rffs)
        self.tu = None

        self.tl_ff = None
        self.tu_ff = None

        # Cache for corresponding properties
        self._tl_traj = None
        self._tu_traj = None

        self._tl_ef = {}
        self._tu_ef = {}

        self.tl_rff = None
        self.tu_rff = None

        # Current time step to display
        self.tcur = None

        self.color_mode = ''

        self.ef_index = None
        self.ef_nt = 100
        self.extremal_fields: Dict[str, Dict[str, Trajectory]] = {}
        self.ef_bulks = {}
        # True or False dict whether group of extremals should be
        # displayed in aggregated mode
        self.ef_agg_display = {}

        self.increment_factor = 0.001

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        # matplotlib.rcParams['text.usetex'] = True

        self.mode_3d = mode_3d

    @staticmethod
    def prompt_data_dir(output_dir: Optional[str] = None,
                        latest=False, last=False) -> str:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'output')) if \
            output_dir is None else output_dir
        cache_fp = os.path.join(output_dir, '.cache_frontend')
        case_dirs = [dir_name for dir_name in os.listdir(output_dir) if
                     os.path.isdir(os.path.join(output_dir, dir_name))
                     and not dir_name.startswith('.')
                     and Display.is_readable(os.path.join(output_dir, dir_name))]
        if len(case_dirs) == 0:
            raise FileNotFoundError('No data found in folder "%s"' % output_dir)
        last_tag = '{LAST}'
        latest_tag = '{LATEST}'
        if latest:
            name = latest_tag
        elif last:
            name = last_tag
        else:
            try:
                import easygui
            except ImportError:
                raise ImportError('"easygui" package required for interactive case selection')
            name = easygui.choicebox('Choose case from available propositions extracted from "%s"' % output_dir,
                                     'Case selection',
                                     [latest_tag, last_tag] + list(sorted(case_dirs)))
        if name is None:
            print('No example selected, quitting', file=sys.stderr)
            exit(1)
        sel_dir = os.path.join(output_dir, name)
        if name == latest_tag:
            print('Opening latest file')
            all_subdirs = list(map(lambda n: os.path.join(output_dir, n), case_dirs))
            all_param_files = [os.path.join(dir_path, 'problem_info.json')
                               for dir_path in all_subdirs if Display.is_readable(dir_path)]

            latest_subdir = os.path.dirname(max(all_param_files, key=os.path.getmtime))
            print(latest_subdir)
            data_dir = latest_subdir
        elif name == last_tag:
            print('Opening last file')
            with open(cache_fp, 'r') as f:
                sel_dir = f.readline()
                print(sel_dir)
            data_dir = sel_dir
        else:
            data_dir = sel_dir
            with open(cache_fp, 'w') as f:
                f.writelines(sel_dir)
        return data_dir

    @classmethod
    def is_readable(cls, dir_path: str):
        params_path = os.path.join(dir_path, 'problem_info.json')
        if os.path.exists(params_path):
            return True
        return False

    @classmethod
    def from_scratch(cls, latest=False, last=False, mode_3d=False):
        case_dir = Display.prompt_data_dir(latest=latest, last=last)
        return cls(os.path.basename(case_dir), case_dir=case_dir, mode_3d=mode_3d)

    @classmethod
    def from_path(cls, path: str, latest=False, last=False, mode_3d=False):
        if Display.is_readable(path):
            case_dir = path
        else:
            case_dir = Display.prompt_data_dir(path, latest=latest, last=last)
        return cls(os.path.basename(case_dir), case_dir=case_dir, mode_3d=mode_3d)

    def configure(self, flags: str):
        self.set_mode(flags)
        self.set_title(os.path.basename(self.case_dir))
        self.load_all()
        self.setup()

    def run(self, noshow=False, movie=False, frames=None, fps=None,
            movie_format='apng', mini=False, flags=''):
        self.configure(flags)
        self.draw_all()
        if movie:
            kwargs = {}
            if frames is not None:
                kwargs['frames'] = frames
            if fps is not None:
                kwargs['fps'] = fps
            kwargs['movie_format'] = movie_format
            kwargs['mini'] = mini
            self.to_movie(**kwargs)
        elif not noshow:
            try:
                self.show()
            except KeyboardInterrupt:
                pass

    @property
    def bl(self):
        return self.io.bl

    @property
    def tr(self):
        return self.io.tr

    @property
    def x_init(self):
        return self.io.x_init

    @property
    def x_target(self):
        return self.io.x_target

    @property
    def target_radius(self):
        return self.io.target_radius

    @property
    def tl_traj(self):
        if self._tl_traj is None:
            val = None
            for traj in self.trajs.values():
                m = np.min(traj.times)
                if val is None or m < val:
                    val = m
            self._tl_traj = val
        return self._tl_traj

    @property
    def tu_traj(self):
        if self._tu_traj is None:
            val = None
            for traj in self.trajs.values():
                m = np.max(traj.times)
                if val is None or m > val:
                    val = m
            self._tu_traj = val
        return self._tu_traj

    def tl_ef(self, ef_id):
        if ef_id not in self._tl_ef.keys():
            val = None
            for traj in self.extremal_fields[ef_id].values():
                m = np.min(traj.times)
                if val is None or m < val:
                    val = m
            self._tl_ef[ef_id] = val
        return self._tl_ef[ef_id]

    def tu_ef(self, ef_id):
        if ef_id not in self._tu_ef.keys():
            val = None
            for traj in self.extremal_fields[ef_id].values():
                m = np.max(traj.times)
                if val is None or m > val:
                    val = m
            self._tu_ef[ef_id] = val
        return self._tu_ef[ef_id]

    def setup(self, debug=False):

        # if self.opti_ceil is None:
        #     self.opti_ceil = 0.0005 * 0.5 * (self.tr[0] - self.bl[0] + self.tr[1] - self.bl[1])

        self.fsc = FontsizeConf()
        plt.rc('font', size=self.fsc.fontsize)
        plt.rc('axes', titlesize=self.fsc.axes_titlesize)
        plt.rc('axes', labelsize=self.fsc.axes_labelsize)
        plt.rc('xtick', labelsize=self.fsc.xtick_labelsize)
        plt.rc('ytick', labelsize=self.fsc.ytick_labelsize)
        plt.rc('legend', fontsize=self.fsc.legend_fontsize)
        # plt.rc('font', family=self.fsc.font_family)
        plt.rc('mathtext', fontset=self.fsc.mathtext_fontset)

        self.main_fig = plt.figure(num=f"dabryvisu ({self.coords.value})",
                                   constrained_layout=False,
                                   figsize=(12, 8))
        self.main_fig.canvas.mpl_disconnect(self.main_fig.canvas.manager.key_press_handler_id)
        self.main_fig.subplots_adjust(
            top=0.93,
            bottom=0.11,
            left=0.1,
            right=0.85,
            hspace=0.155,
            wspace=0.13
        )
        self.main_fig.suptitle(self.title)
        if self.mode_3d:
            self.main_ax = self.main_fig.add_subplot(projection='3d')
        else:
            self.main_ax = self.main_fig.add_subplot(box_aspect=1., anchor='C')

        self.setup_map()
        # self.setup_components()
        self.display_setup = True

        # self.ax_rbutton = self.mainfig.add_axes([0.44, 0.025, 0.08, 0.05])
        # self.reload_button = Button(self.ax_rbutton, 'Reload', color='white',
        #                             hovercolor='grey')
        # self.reload_button.label.set_fontsize(self.fsc.button_fontsize)
        # self.reload_button.on_clicked(lambda event: self.reload())
        #
        # self.ax_sbutton = self.mainfig.add_axes([0.34, 0.025, 0.08, 0.05])
        # self.switch_button = Button(self.ax_sbutton, 'Switch', color='white',
        #                             hovercolor='grey')
        # self.switch_button.label.set_fontsize(self.fsc.button_fontsize)
        # self.switch_button.on_clicked(lambda event: self.switch_agg())

        # self.ax_cbutton = self.mainfig.add_axes([0.54, 0.025, 0.08, 0.05])
        # self.control_button = CheckButtons(self.ax_cbutton, ['Controls'], [False])
        # self.control_button.labels[0].set_fontsize(self.fsc.button_fontsize)
        # self.control_button.on_clicked(lambda event: self.toggle_controls())

        self.ax_info = self.main_fig.text(0.34, 0.025, ' ')

        self.setup_slider()

        self.leg_handles = []
        self.leg_labels = []

    def setup_slider(self):
        self.ax_timeslider = self.main_fig.add_axes([0.03, 0.25, 0.0225, 0.63])
        self.ax_timedisplay = self.main_fig.text(0.03, 0.04, f'', fontsize=self.fsc.timedisp_major)
        self.ax_timedisp_minor = self.main_fig.text(0.03, 0.018, f'', fontsize=self.fsc.timedisp_minor)
        val_init = 1.
        self.time_slider = Slider(
            ax=self.ax_timeslider,
            label="Time",
            valmin=0.,
            valmax=1.,
            valinit=val_init,
            orientation="vertical"
        )
        try:
            self.reload_time(val_init)
        except TypeError:
            pass
        self.time_slider.on_changed(self.reload_time)

    def setup_map(self):
        """
        Sets the display of the map
        """
        cartesian = (self.coords == Coords.CARTESIAN)
        gcs = (self.coords == Coords.GCS)

        if self.mode_3d:
            self.ax = self.main_ax
            bl_display = self.bl - np.diag((self.x_offset, self.y_offset)) @ (self.tr - self.bl)
            tr_display = self.tr + np.diag((self.x_offset, self.y_offset)) @ (self.tr - self.bl)
            self.ax.set_xlim(bl_display[0], tr_display[0])
            self.ax.set_ylim(bl_display[1], tr_display[1])
            self.ax.set_zlim(0, self.engy_max)
            return

        if gcs:
            kwargs = {
                'resolution': 'l',
                'projection': self.projection,
                'ax': self.main_ax
            }
            # Don't plot coastal lines features less than 1000km^2
            # kwargs['area_thresh'] = (6400e3 * np.pi / 180) ** 2 * (self.tr[0] - self.bl[0]) * 0.5 * (
            #             self.bl[1] + self.tr[1]) / 1000.

            if self.projection == 'merc':
                kwargs['llcrnrlon'] = Utils.RAD_TO_DEG * (self.bl[0] - self.x_offset * (self.tr[0] - self.bl[0]))
                kwargs['llcrnrlat'] = Utils.RAD_TO_DEG * (self.bl[1] - self.y_offset * (self.tr[1] - self.bl[1]))
                kwargs['urcrnrlon'] = Utils.RAD_TO_DEG * (self.tr[0] + self.x_offset * (self.tr[0] - self.bl[0]))
                kwargs['urcrnrlat'] = Utils.RAD_TO_DEG * (self.tr[1] + self.y_offset * (self.tr[1] - self.bl[1]))

            elif self.projection == 'ortho':
                kwargs['lon_0'], kwargs['lat_0'] = \
                    0.5 * Utils.RAD_TO_DEG * (np.array((self.bl[0], self.bl[1])) + np.array((self.tr[0], self.tr[1])))
                # tuple(Utils.RAD_TO_DEG * np.array(middle(np.array((self.bl[0], self.bl[1])),
                #                                    np.array((self.tr[0], self.tr[1])))))
                proj = Proj(proj='ortho', lon_0=kwargs['lon_0'], lat_0=kwargs['lat_0'])
                # pgrid = np.array(proj(Utils.RAD_TO_DEG * self.ff['grid'][:, :, 0],
                # Utils.RAD_TO_DEG * self.ff['grid'][:, :, 1]))
                # px_min = np.min(pgrid[0])
                # px_max = np.max(pgrid[0])
                # py_min = np.min(pgrid[1])
                # py_max = np.max(pgrid[1])
                if self.x_init is not None and self.x_target is not None:
                    x1, y1 = proj(*(Utils.RAD_TO_DEG * self.x_init))
                    x2, y2 = proj(*(Utils.RAD_TO_DEG * self.x_target))
                    x_min = min(x1, x2)
                    x_max = max(x1, x2)
                    y_min = min(y1, y2)
                    y_max = max(y1, y2)

                    kwargs['llcrnrx'], kwargs['llcrnry'] = \
                        x_min - self.x_offset_gcs * (x_max - x_min), \
                        y_min - self.y_offset_gcs * (y_max - y_min)
                    kwargs['urcrnrx'], kwargs['urcrnry'] = \
                        x_max + self.x_offset_gcs * (x_max - x_min), \
                        y_max + self.y_offset_gcs * (y_max - y_min)
                    bounds1 = proj(kwargs['llcrnrx'], kwargs['llcrnry'])
                    bounds2 = proj(kwargs['urcrnrx'], kwargs['urcrnry'])
                    for b in bounds1 + bounds2:
                        if np.isnan(b):
                            del kwargs['llcrnrx']
                            del kwargs['llcrnry']
                            del kwargs['urcrnrx']
                            del kwargs['urcrnry']
                            break
            elif self.projection == 'omerc':
                kwargs['lon_1'] = Utils.RAD_TO_DEG * self.x_init[0]
                kwargs['lat_1'] = Utils.RAD_TO_DEG * self.x_init[1]
                kwargs['lon_2'] = Utils.RAD_TO_DEG * self.x_target[0]
                kwargs['lat_2'] = Utils.RAD_TO_DEG * self.x_target[1]

                def middle(x1, x2):
                    bx = np.cos(x2[1]) * np.cos(x2[0] - x1[0])
                    by = np.cos(x2[1]) * np.sin(x2[0] - x1[0])
                    return x1[0] + atan2(by, np.cos(x1[1]) + bx), \
                           atan2(np.sin(x1[1]) + np.sin(x2[1]), np.sqrt((np.cos(x1[1]) + bx) ** 2 + by ** 2))

                def distance(x1, x2):
                    g = Geod(ellps='WGS84')
                    (az12, az21, dist) = g.inv(x1[0], x1[1], x2[0], x2[1], radians=True)
                    return dist

                lon_0, lat_0 = middle(self.x_init, self.x_target)
                kwargs['lon_0'], kwargs['lat_0'] = Utils.RAD_TO_DEG * lon_0, Utils.RAD_TO_DEG * lat_0
                kwargs['width'] = distance(self.x_init, self.x_target) * 1.35
                kwargs['height'] = distance(self.x_init, self.x_target) * 1.35

            elif self.projection == 'lcc':
                kwargs['lon_0'] = Utils.RAD_TO_DEG * 0.5 * (self.bl[0] + self.tr[0])
                kwargs['lat_0'] = Utils.RAD_TO_DEG * 0.5 * (self.bl[1] + self.tr[1])
                kwargs['lat_1'] = Utils.RAD_TO_DEG * (self.tr[1] + self.y_offset * (self.tr[1] - self.bl[1]))
                kwargs['lat_2'] = Utils.RAD_TO_DEG * (self.bl[1] - self.y_offset * (self.tr[1] - self.bl[1]))
                kwargs['width'] = (1 + self.x_offset) * (self.tr[0] - self.bl[0]) * Utils.EARTH_RADIUS
                kwargs['height'] = kwargs['width']

            else:
                print(f'Projection type "{self.projection}" not handled yet', file=sys.stderr)
                exit(1)

            self.map = Basemap(**kwargs)

            # self.map.shadedrelief()
            self.map.drawcoastlines()
            self.map.fillcontinents()
            if self.x_init is not None and self.x_target is not None:
                self.map.drawgreatcircle(Utils.RAD_TO_DEG * self.x_init[0],
                                         Utils.RAD_TO_DEG * self.x_init[1],
                                         Utils.RAD_TO_DEG * self.x_target[0],
                                         Utils.RAD_TO_DEG * self.x_target[1], color='grey', linestyle='--')
            lw = 0.5
            dashes = (2, 2)
            # draw parallels
            try:
                lat_min = Utils.RAD_TO_DEG * min(self.bl[1], self.tr[1])
                lat_max = Utils.RAD_TO_DEG * max(self.bl[1], self.tr[1])
                n_lat = floor((lat_max - lat_min) / 10) + 2
                self.map.drawparallels(10. * (floor(lat_min / 10.) + np.arange(n_lat)),  # labels=[1, 0, 0, 0],
                                       linewidth=lw,
                                       dashes=dashes, textcolor=(1., 1., 1., 0.))
                # draw meridians
                lon_min = Utils.RAD_TO_DEG * min(self.bl[0], self.tr[0])
                lon_max = Utils.RAD_TO_DEG * max(self.bl[0], self.tr[0])
                n_lon = floor((lon_max - lon_min) / 10) + 2
                self.map.drawmeridians(10. * (floor(lon_min / 10.) + np.arange(n_lon)),  # labels=[1, 0, 0, 1],
                                       linewidth=lw,
                                       dashes=dashes)
            except ValueError:
                pass

        if cartesian:
            if self.axes_equal:
                self.main_ax.axis('equal')
            self.main_ax.set_xlim(self.bl[0] - self.x_offset * (self.tr[0] - self.bl[0]),
                                  self.tr[0] + self.x_offset * (self.tr[0] - self.bl[0]))
            self.main_ax.set_ylim(self.bl[1] - self.y_offset * (self.tr[1] - self.bl[1]),
                                  self.tr[1] + self.y_offset * (self.tr[1] - self.bl[1]))
            self.main_ax.set_xlabel('$x_1$')
            self.main_ax.set_ylabel('$x_2$')
            self.main_ax.grid(visible=True, linestyle='-.', linewidth=0.5)
            self.main_ax.tick_params(direction='in')
            formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits([-3, 4])
            self.main_ax.xaxis.set_major_formatter(formatter)
            self.main_ax.yaxis.set_major_formatter(formatter)

        if cartesian:
            self.ax = self.main_ax
        if gcs:
            self.ax = self.map

    def draw_point(self, x, y, label=None):
        kwargs = {
            's': 8.,
            'color': 'red',
            'marker': 'D',
            'zorder': ZOrder.ANNOT.value
        }
        if self.coords == Coords.GCS:
            kwargs['latlon'] = True
            pos_annot = self.map(x, y)
        else:
            pos_annot = (x, y)
        self.ax.scatter(x, y, **kwargs)
        if label is not None:
            self.ax.annotate(label, pos_annot, (10, 10), textcoords='offset pixels', ha='center')

    def clear_ff(self):
        if self.ff_colormesh is not None:
            self.ff_colormesh.remove()
            self.ff_colormesh = None
        if self.ff_quiver is not None:
            self.ff_quiver.remove()
            self.ff_quiver = None
        # if self.ff_colorbar is not None:
        #     self.ff_colorbar.remove()
        #     self.ff_colorbar = None

    def clear_trajs(self):
        # if len(self.traj_lp) + len(self.traj_lines) + len(self.traj_ticks) + len(self.traj_controls) > 0:
        for l in self.traj_lines:
            for a in l:
                a.remove()
        for a in self.traj_epoints:
            a.remove()
        for a in self.traj_lp:
            a.remove()
        for a in self.traj_annot:
            a.remove()
        for a in self.traj_ticks:
            a.remove()
        for a in self.traj_controls:
            a.remove()
        self.traj_lines = []
        self.traj_epoints = []
        self.traj_ticks = []
        self.traj_annot = []
        self.traj_lp = []
        self.traj_controls = []
        self.id_traj_color = 0

    def clear_rff(self):
        if len(self.rff_contours) != 0:
            for c in self.rff_contours:
                for coll in c.collections:
                    coll.remove()
            self.rff_contours = []

    def clear_obs(self):
        if len(self.obs_contours) != 0:
            for c in self.obs_contours:
                for coll in c.collections:
                    coll.remove()
            self.obs_contours = []

    def clear_pen(self):
        if len(self.pen_contours) != 0:
            for c in self.pen_contours:
                for coll in c.collections:
                    coll.remove()
            self.pen_contours = []

    def clear_solver(self):
        if self.scatter_init is not None:
            self.scatter_init.remove()
        if self.scatter_target is not None:
            self.scatter_target.remove()
        if self.circle_target is not None:
            self.circle_target.remove()

    def load_filter(self):
        self.filter_fpath = os.path.join(self.case_dir, self.filter_fname)
        if not os.path.exists(self.filter_fpath):
            return
        with open(self.filter_fpath, 'r') as f:
            self.traj_filter = list(map(str.strip, f.readlines()))

    def load_ff(self):
        self.ff = DiscreteFF.from_npz(self.io.ff_fpath, no_diff=True)
        norms = np.linalg.norm(self.ff.values, axis=-1)
        self.ff_norm_min, self.ff_norm_avg, self.ff_norm_max = np.min(norms), np.average(norms), np.max(norms)
        if self.ff.is_unsteady:
            self.tl_ff = self.ff.t_start
            self.tu_ff = self.ff.t_end
            ut = 1
            if self.rescale_ff:
                nt, nx, ny, _ = self.ff.values.shape
                nxp, nyp = nx, ny
                while nt * nxp * nyp * 2 > 500000:
                    ut += 1
                    nxp = 1 + (nx - 1) // ut
                    nyp = 1 + (ny - 1) // ut
            nt = 1000
            times = np.linspace(self.ff.times[0], self.ff.times[-1], nt)
            values = np.zeros((nt, *self.ff.values[:, ::ut, ::ut, :].shape[1:]))
            for it, t in enumerate(times):
                k, p = index(t, self.ff.times)
                values[it, ...] = (1 - p) * self.ff.values[k, ::ut, ::ut, :] + p * self.ff.values[k + 1, ::ut, ::ut, :]
            self.ff_display = DiscreteFF(values, self.ff.bounds, self.ff.coords, no_diff=True)
        else:
            values = np.expand_dims(self.ff.values, 0)
            bounds = np.vstack((np.array((0, 0)), self.ff.bounds))
            self.ff_display = DiscreteFF(values, bounds, self.ff.coords, no_diff=True)

    def load_trajs(self, filename=None):
        self.trajs.clear()
        if not os.path.exists(self.io.trajs_dir):
            return
        for filename in os.listdir(self.io.trajs_dir):
            fpath = os.path.join(self.io.trajs_dir, filename)
            if fpath.endswith('.json'):
                continue
            if not os.path.isdir(fpath):
                traj = Trajectory.from_npz(fpath)
                name = traj_filepath_to_name(fpath)
                self.trajs[name] = traj
                self.trajs_regular[name] = traj
            else:
                collection_name = os.path.basename(fpath)
                if collection_name.startswith('ef'):
                    self.extremal_fields[collection_name] = {}
                for ffname in os.listdir(fpath):
                    ffpath = os.path.join(fpath, ffname)
                    if ffpath.endswith('.json'):
                        continue
                    traj = Trajectory.from_npz(ffpath)
                    name = collection_name + '_' + traj_filepath_to_name(ffpath)
                    self.trajs[name] = traj
                    if collection_name.startswith('ef'):
                        self.extremal_fields[collection_name][name] = traj
                    else:
                        self.trajs_regular[name] = traj
        for ef_name, ef_dict in self.extremal_fields.items():
            n_trajs = len(ef_dict)
            t_start = None
            t_end = None
            dt = None
            for traj in ef_dict.values():
                t_start_cand = traj.times[0]
                t_end_cand = traj.times[-1]
                if len(traj) >= 2:
                    dt = traj.times[1] - traj.times[0]
                if t_start is None or t_start_cand < t_start:
                    t_start = t_start_cand
                if t_end is None or t_end_cand > t_end:
                    t_end = t_end_cand
            nt = int(np.round((t_end - t_start) / dt)) + 1
            bulk = np.nan * np.ones((nt, n_trajs, 2))
            times = np.linspace(t_start, t_end, nt)
            for i_traj, traj in enumerate(ef_dict.values()):
                i_start = (np.abs(times - traj.times[0])).argmin()
                try:
                    bulk[i_start:i_start + len(traj), i_traj, :] = traj.states[:]
                except ValueError:
                    exit(1)
            self.ef_bulks[ef_name] = bulk

    def load_rff(self, filename=None):
        self.rff = None
        self.rff_zero_ceils = []
        if self.rff_fpath is None:
            filename = self.rff_fname if filename is None else filename
            self.rff_fpath = os.path.join(self.case_dir, filename)
        if not os.path.exists(self.rff_fpath):
            return
        with h5py.File(self.rff_fpath, 'r') as f:
            if self.coords != Coords.from_string(f.attrs['coords']):
                warnings.warn(
                    f'RFF coords "{f.attrs["coords"]}" does not match current display coords "{self.coords.value}"',
                    category=UserWarning)

            if self.coords == Coords.GCS:
                self.rff_cntr_kwargs['latlon'] = True
            self.nt_rft, self.nx_rft, self.ny_rft = f['data'].shape
            # ceil = min((self.tr[0] - self.bl[0]) / (3 * self.nx_rft),
            #            (self.tr[1] - self.bl[1]) / (3 * self.ny_rft))
            nt = self.nt_rft
            self.rff_zero_ceils = []
            # if 'nt_rft_eff' in self.__dict__:
            #     nt = self.nt_rft_eff + 1
            # else:
            self.rff = {}
            failed_zeros = []
            failed_ceils = []
            self.rff = {}
            self.rff['data'] = np.zeros(f['data'].shape)
            self.rff['data'][:] = f['data']
            self.rff['grid'] = np.zeros(f['grid'].shape)
            self.rff['grid'][:] = f['grid']
            self.rff['ts'] = np.zeros(f['ts'].shape)
            self.rff['ts'][:] = f['ts']

            for k in range(nt):

                self.tl_rff = self.rff['ts'][0]
                self.tu_rff = self.rff['ts'][-1]

                data_max = self.rff['data'][k, :, :].max()
                data_min = self.rff['data'][k, :, :].min()

                zero_ceil = min((self.rff['grid'][:, :, 0].max() - self.rff['grid'][:, :, 0].min()) / (3 * self.nx_rft),
                                (self.rff['grid'][:, :, 1].max() - self.rff['grid'][:, :, 1].min()) / (3 * self.ny_rft))
                self.rff_zero_ceils.append(zero_ceil)

                if data_min > zero_ceil or data_max < -zero_ceil:
                    failed_zeros.append(k)

                # Adjust zero ceil if needed
                absvals = np.sort(np.abs(self.rff['data'].flatten()))
                # Take fifth pencent quantile value
                absceil = absvals[int(0.01 * absvals.shape[0])]
                if absceil > self.rff_zero_ceils[k] / 2:
                    failed_ceils.append(k)
                    self.rff_zero_ceils[k] = absceil

            if len(failed_ceils) > 0:
                Display._info(f'Total {len(failed_ceils)} FF needed ceil adjustment. {tuple(failed_ceils)}')

            if len(failed_zeros) > 0:
                warnings.warn(f'No RFF value in zero band for indexes {tuple(failed_zeros)}', category=UserWarning)

    def load_obs(self):
        self.obstacles = []
        if not os.path.exists(self.io.obs_dir):
            return
        for obs_fpath in list(map(lambda path: os.path.join(self.io.obs_dir, path), os.listdir(self.io.obs_dir))):
            self.obstacles.append(DiscreteObs.from_npz(obs_fpath, no_diff=True))
        self.obs_total_values = np.min(np.stack([obs.values for obs in self.obstacles], -1), axis=-1)
        obs = self.obstacles[0]
        self.obs_grid = np.stack(np.meshgrid(
            np.linspace(obs.bounds[0, 0], obs.bounds[0, 1], obs.values.shape[0]),
            np.linspace(obs.bounds[1, 0], obs.bounds[1, 1], obs.values.shape[1]), indexing='ij'), -1)

    def load_pen(self, filename=None):
        if self.pen_fpath is None:
            filename = self.pen_fname if filename is None else filename
            self.pen_fpath = os.path.join(self.case_dir, filename)
        if not os.path.exists(self.pen_fpath):
            return
        with h5py.File(self.pen_fpath, 'r') as f:
            self.penalty = {}
            self.penalty['data'] = np.zeros(f['data'].shape)
            self.penalty['data'][:] = f['data']
            self.penalty['grid'] = np.zeros(f['grid'].shape)
            self.penalty['grid'][:] = f['grid']
            self.penalty['ts'] = np.zeros(f['ts'].shape)
            self.penalty['ts'][:] = f['ts']

    def load_all(self):
        self.load_filter()
        self.load_ff()
        self.load_trajs()
        self.load_rff()
        self.load_obs()
        self.load_pen()
        self.adapt_tw()
        n_trajs = len(self.trajs)
        n_rffs = 0 if self.rff is None else self.rff['data'].shape[0]
        if self.tl is not None and self.tu is not None:
            self.tcur = 0.5 * (self.tl + self.tu)
        self.preprocessing()
        n_ef = len(self.extremal_fields.keys())
        n_eft = sum((len(tgv) for tgv in self.extremal_fields.values()))
        Display._info(
            f'Loading completed. {n_trajs - n_eft} regular trajs, '
            f'{n_ef} extremal fields of {n_eft} trajs, '
            f'{len(self.obstacles)} obstacles.')

    def adapt_tw(self):
        """
        Adapt time window to data
        """

        def mysat(*args, minimize=False):
            all_none = True
            values = []
            for a in args:
                if a is not None:
                    all_none = False
                    values.append(a)
            if all_none:
                raise Exception('All values are None for saturation function')
            return min(values) if minimize else max(values)

        if self.tl_traj is None and self.tl_rff is None:
            self.tl = self.tl_ff
            self.tu = self.tu_ff
        else:
            self.tl = mysat(self.tl_traj, self.tl_rff, minimize=True)
            self.tu = mysat(self.tu_traj, self.tu_rff, minimize=False)

    def preprocessing(self):
        # Preprocessing
        # For extremal fields, group points in fronts
        if len(self.extremal_fields) == 0:
            return
        ef_0_name = list(self.extremal_fields.keys())[0]
        self.ef_index = lambda t: None if t < self.tl_ef(ef_0_name) or \
                                          t > self.tu_ef(ef_0_name) else \
            floor((self.ef_nt - 1) * (t - self.tl_ef(ef_0_name)) / (self.tu_ef(ef_0_name) - self.tl_ef(ef_0_name)))
        no_display = False
        for ef_id in self.extremal_fields.keys():
            if not no_display:
                self.ef_agg_display[ef_id] = True
                no_display = True
            else:
                self.ef_agg_display[ef_id] = False

    def draw_rff(self, debug=False, interp=True):
        self.clear_rff()
        if self.rff is not None:
            ax = None
            if debug:
                fig, ax = plt.subplots()
            nt = self.rff['data'].shape[0]

            if interp == False:
                il = 0
                iu = nt
                ts = self.rff['ts']
                at_least_one = False
                for i in range(ts.shape[0]):
                    if ts[i] < self.tl:
                        il += 1
                    elif ts[i] > self.tcur:
                        iu = i - 1
                        break
                    else:
                        at_least_one = True
                if not at_least_one:
                    return
                if iu <= il:
                    return

                for k in range(il, iu):
                    data_max = self.rff['data'][k, :, :].max()
                    data_min = self.rff['data'][k, :, :].min()
                    if debug:
                        ax.hist(self.rff['data'][k, :, :].reshape(-1), density=True, label=k,
                                color=path_colors[k % len(path_colors)])
                    zero_ceil = self.rff_zero_ceils[k]  # (data_max - data_min) / 1000.
                    if debug:
                        print(f'{k}, min : {data_min}, max : {data_max}, zero_ceil : {zero_ceil}')
                    factor = Utils.RAD_TO_DEG if self.coords == Coords.GCS else 1.
                    args = (factor * self.rff['grid'][:, :, 0],
                            factor * self.rff['grid'][:, :, 1],
                            self.rff['data'][k, :, :]) + (
                               ([-zero_ceil / 2, zero_ceil / 2],) if not debug else ([data_min, 0., data_max],))
                    # ([-1000., 1000.],) if not debug else (np.linspace(-100000, 100000, 200),))
                    self.rff_contours.append(self.ax.contourf(*args, **self.rff_cntr_kwargs))
                if debug:
                    ax.legend()
                    plt.show()
            else:
                i, alpha = index(self.tcur, self.rff['ts'])
                rff_values = (1 - alpha) * self.rff['data'][i, :, :] + alpha * self.rff['data'][i + 1, :, :]
                zero_ceil = (1 - alpha) * self.rff_zero_ceils[i] + alpha * self.rff_zero_ceils[i + 1]
                if self.coords == Coords.GCS:
                    nx, ny = self.rff['grid'][:, :, 0].shape
                    xi = np.linspace(self.rff['grid'][:, :, 0].min(), self.rff['grid'][:, :, 0].max(), 4 * nx)
                    yi = np.linspace(self.rff['grid'][:, :, 1].min(), self.rff['grid'][:, :, 1].max(), 4 * ny)
                    xi, yi = np.meshgrid(xi, yi)

                    # interpolate
                    x, y, z = self.rff['grid'][:, :, 0], self.rff['grid'][:, :, 1], rff_values
                    zi = griddata((x.reshape(-1), y.reshape(-1)), z.reshape(-1), (xi, yi))
                    args = (Utils.RAD_TO_DEG * xi, Utils.RAD_TO_DEG * yi, zi)
                else:
                    args = (self.rff['grid'][:, :, 0], self.rff['grid'][:, :, 1], rff_values)
                args = args + ([-zero_ceil / 2, zero_ceil / 2],)
                # ([-1000., 1000.],) if not debug else (np.linspace(-100000, 100000, 200),))
                self.rff_contours.append(self.ax.contourf(*args, **self.rff_cntr_kwargs))

    def draw_ff(self):
        if np.isclose(self.ff_norm_max, 0):
            return
        self.clear_ff()
        if self.ff_display.is_unsteady:
            it, _ = index(self.tcur, self.ff_display.times)
            ff_values = self.ff_display.values[it, ...]
        else:
            ff_values = self.ff_display.values
        factor = Utils.RAD_TO_DEG if self.coords == Coords.GCS else 1.
        grid = np.stack(
            np.meshgrid(np.linspace(self.ff_display.bounds[-2, 0], self.ff_display.bounds[-2, 1],
                                    self.ff_display.values.shape[-3]),
                        np.linspace(self.ff_display.bounds[-1, 0], self.ff_display.bounds[-1, 1],
                                    self.ff_display.values.shape[-2]),
                        indexing='ij'), -1)

        norm = mpl_colors.Normalize()
        self.cm = self.selected_cm
        if self.coords == Coords.GCS and self.ff_norm_max < 1.5 * self.cm_norm_max and \
                self.ff_norm_avg > self.cm_norm_min_mag:
            self.cm = windy_cm
            norm.autoscale(np.array([self.cm_norm_min, self.cm_norm_max]))
        else:
            self.cm = jet_desat_cm
            norm.autoscale(np.array([self.ff_norm_min, self.ff_norm_max]))

        needs_engy = self.mode_energy and self.mode_ef and self.mode_aggregated
        set_engycb = needs_engy and self.active_ffcb
        if self.sm_ff is None:
            self.sm_ff = mpl_cm.ScalarMappable(cmap=self.cm, norm=norm)
            if self.engy_min and self.engy_max is not None:
                self.sm_engy = mpl_cm.ScalarMappable(cmap='tab20b',
                                                     norm=mpl_colors.Normalize(
                                                         vmin=self.engy_min / 3.6e6,
                                                         vmax=self.engy_max / 3.6e6))
            if self.coords == Coords.GCS:
                self.ff_colorbar = self.ax.colorbar(self.sm_ff, pad=0.1)
            elif self.coords == Coords.CARTESIAN:
                self.ff_colorbar = self.main_fig.colorbar(self.sm_ff, ax=self.ax, pad=0.03)
            self.ff_colorbar.set_label('Flow field', labelpad=10)
            self.active_ffcb = True

        set_ffcb = not needs_engy and not self.active_ffcb
        if set_ffcb:
            self.ff_colorbar.update_normal(self.sm_ff)
            self.ff_colorbar.set_label('Flow field')
            self.active_ffcb = True
        elif set_engycb and self.sm_engy is not None:
            self.ff_colorbar.update_normal(self.sm_engy)
            self.ff_colorbar.set_label('Energy [kWh]')
            self.active_ffcb = False

        kwargs = {
            'cmap': self.cm,
            'norm': norm,
            'alpha': 0.7,
            'zorder': ZOrder.WIND_NORM.value,
            'shading': 'auto',
            'antialiased': True,
        }
        if self.coords == Coords.GCS:
            kwargs['latlon'] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ff_colormesh = self.ax.pcolormesh(
                factor * grid[:, :, 0],
                factor * grid[:, :, 1], np.linalg.norm(ff_values, axis=-1), **kwargs)

        kwargs = {
            'color': (0.2, 0.2, 0.2, 1.0),
            # 'width': 1,  # 0.001,
            'pivot': 'tail',
            'alpha': 0.7,
            'units': 'xy',
            'minshaft': 2.,
            'zorder': ZOrder.WIND_VECTORS.value
        }
        if self.coords == Coords.GCS:
            kwargs['latlon'] = True
        # if np.any(qnorms > self.cm_norm_min + 0.01 * (self.cm_norm_max - self.cm_norm_min)):
        self.ff_quiver = self.ax.quiver(factor * grid[:, :, 0], factor * grid[:, :, 1],
                                        ff_values[..., 0], ff_values[..., 1], **kwargs)

    def draw_trajs(self):
        self.clear_trajs()
        for name in self.trajs_regular.keys():
            self.draw_traj(name, showlabel=True)
        if self.mode_ef:
            self.draw_ef()

    def draw_ef(self):
        if not self.mode_aggregated:
            if self.ef_index is not None:
                # The time cursor corresponds to given index in front list
                i = self.ef_index(self.tcur)
                if i is not None:
                    for ef_id, bulks in self.ef_bulks.items():
                        try:
                            points = bulks[i]
                        except IndexError:
                            continue
                        # Last points
                        kwargs = {'s': 5.,
                                  'color': reachability_colors['pmp']['last'],
                                  'marker': 'o',
                                  'linewidths': 1.,
                                  'zorder': ZOrder.TRAJS.value,
                                  }
                        if self.coords == Coords.GCS:
                            kwargs['latlon'] = True
                            points = np.dot(points, np.diag((Utils.RAD_TO_DEG, Utils.RAD_TO_DEG)))
                        if self.mode_3d:
                            kwargs = {
                                'color': reachability_colors['pmp']['last'],
                                'zorder': ZOrder.TRAJS.value
                            }
                            scatter = self.ax.scatter(points[..., 0], points[..., 1], points[..., 2], **kwargs)
                        else:
                            scatter = self.ax.scatter(points[..., 0], points[..., 1], **kwargs)
                        self.traj_lp.append(scatter)
        else:
            for ef_name in self.extremal_fields.keys():
                if not self.ef_agg_display[ef_name]:
                    continue
                for name, traj in self.extremal_fields[ef_name].items():
                    self.draw_traj(name, ef_id=ef_name)

    def draw_traj(self, name, ef_id=None, showlabel=False):
        """
        Plots the given trajectory according to selected display mode
        """
        # duration = (ts[last_index - 1] - ts[0])
        # self.t_tick = max_time / (nt_tick - 1)
        trajs = self.trajs if ef_id is None else self.extremal_fields[ef_id]

        linewidth = None
        if ef_id is not None:
            linewidth = 1.5

        # Color selection
        color = {}
        if ef_id is None:
            c = path_colors[list(trajs.keys()).index(name) % len(path_colors)]
            color['steps'] = c
            color['last'] = c
        else:
            color['steps'] = reachability_colors['pmp']['steps']
            color['last'] = reachability_colors['pmp']['last']

        # ls = 'solid' if 'm1' not in info else '--'  # linestyle[label % len(linestyle)]
        # ls = '--' if 'optimal' in _type else ls
        # ls = 'dashdot' if 'optimal' in _type and 'm1' in info else ls

        kwargs = {
            'color': color['steps'],
            'linestyle': 'solid',
            'label': name if showlabel else None,
            'zorder': ZOrder.TRAJS.value,
            'alpha': 0.7,
            'linewidth': 2.5 if linewidth is None else linewidth,
        }
        points = trajs[name].states[trajs[name].times < self.tcur]
        if self.coords == Coords.GCS:
            kwargs['latlon'] = True
            points = np.dot(points, np.diag((Utils.RAD_TO_DEG, Utils.RAD_TO_DEG)))

        if ef_id is None or self.mode_ef_display:
            self.traj_lines.append(self.ax.plot(points[..., 0], points[..., 1], **kwargs))

        if self.mode_energy:
            c = trajs[name].cost[trajs[name].times < self.tcur]
            # norm = mpl_colors.Normalize(vmin=3.6e6, vmax=10*3.6e6)
            self.traj_epoints.append(
                self.ax.scatter(points[:-1, 0][::-1], points[:-1, 1][::-1], c=c[::-1], cmap='tab20b',
                                norm=self.engy_norm))

        # Last points
        kwargs = {'s': 7.,
                  'color': color['last'],
                  'marker': 'o',
                  'linewidths': 1.,
                  'zorder': ZOrder.TRAJS.value,
                  # 'label': info,
                  }
        if points.size > 0:
            px = points[-1, 0]
            py = points[-1, 1]
            if self.coords == Coords.GCS:
                kwargs['latlon'] = True
                px = Utils.RAD_TO_DEG * px
                py = Utils.RAD_TO_DEG * py
            scatter = self.ax.scatter(px, py, **kwargs)
            self.traj_lp.append(scatter)

            # Annotation
            if self.mode_annot:
                if self.coords == Coords.CARTESIAN:
                    self.traj_annot.append(self.ax.annotate(name, xy=(px, py), fontsize='x-small'))
                else:

                    self.traj_annot.append(self.main_ax.annotate(name, xy=self.map(px, py), fontsize='x-small'))

        # Heading vectors
        factor = 1. if self.coords == Coords.CARTESIAN else Utils.EARTH_RADIUS
        kwargs = {
            'color': (0.2, 0.2, 0.2, 1.0),
            'pivot': 'tail',
            'alpha': 1.0,
            'zorder': ZOrder.WIND_VECTORS.value
        }
        if self.coords == Coords.GCS:
            kwargs['latlon'] = True
            kwargs['width'] = 1 / 500  # factor ** 2 / 1000000
            kwargs['scale'] = 50
            # kwargs['scale'] = 1 / factor
            # kwargs['units'] = 'xy'
        elif self.coords == Coords.CARTESIAN:
            kwargs['width'] = 1 / 500
            kwargs['scale'] = 50
        u = 0  # u = controls[iu]
        # f = -1. if ts[1] < ts[0] else 1.
        # cvec = f * np.array((np.cos(u), np.sin(u)))
        # if self.coords == Coords.GCS:
        #     cvec = cvec[::-1]
        # if self.mode_controls:
        #     self.traj_controls.append(self.ax.quiver(px, py, cvec[0], cvec[1], **kwargs))

    def draw_solver(self, labeling=True):
        self.clear_solver()
        kwargs = {}
        # Selecting correct plot axis
        if self.coords == Coords.GCS:
            kwargs['latlon'] = True
            scatterax = self.map
        else:
            scatterax = self.main_ax

        # Init point
        if self.x_init is not None:
            factor = Utils.RAD_TO_DEG if self.coords == Coords.GCS else 1.
            kwargs['s'] = 100 if self.coords == Coords.GCS else 70
            kwargs['color'] = 'black'
            kwargs['marker'] = 'o'
            kwargs['zorder'] = ZOrder.ANNOT.value
            self.scatter_init = scatterax.scatter(*(factor * self.x_init), **kwargs)

            # if labeling:
            #     self.mainax.annotate('Init', c, (10, 10), textcoords='offset pixels', ha='center')
            # if target_radius is not None:
            #     self.mainax.add_patch(plt.Circle(c, target_radius))
        # Target point
        if self.x_target is not None:
            factor = Utils.RAD_TO_DEG if self.coords == Coords.GCS else 1.
            kwargs['s'] = 230 if self.coords == Coords.GCS else 100
            kwargs['color'] = 'black'
            kwargs['marker'] = '*'
            kwargs['zorder'] = ZOrder.ANNOT.value
            self.scatter_target = scatterax.scatter(*(factor * self.x_target), **kwargs)

            # if labeling:
            #     self.mainax.annotate('Target', c, (10, 10), textcoords='offset pixels', ha='center')
            if self.coords == Coords.CARTESIAN:
                ax_circle = scatterax
                pos_center = self.x_target
                radius = self.target_radius
            else:
                ax_circle = self.main_ax
                pos_center = self.map(*(factor * self.x_target))
                radius = self.map(*(factor * (self.x_target + self.target_radius * np.array((1, 0)))))[0] - pos_center[
                    0]
            self.circle_target = ax_circle.add_patch(plt.Circle(pos_center, radius,
                                                                facecolor='none', edgecolor='black'))

    def draw_obs(self):
        self.clear_obs()
        if len(self.obstacles) == 0:
            return
        if self.coords == Coords.CARTESIAN:
            ax = self.main_ax
            kwargs = {}
            factor = 1.
        else:
            ax = self.ax
            kwargs = {'latlon': True}
            factor = Utils.RAD_TO_DEG
        matplotlib.rcParams['hatch.color'] = (.2, .2, .2, 1.)
        self.obs_contours.append(ax.contourf(factor * self.obs_grid[..., 0],
                                             factor * self.obs_grid[..., 1],
                                             self.obs_total_values,
                                             alpha=0.,
                                             levels=[-1, 0],
                                             extend='min',
                                             # colors='grey',
                                             hatches=['//'],
                                             antialiased=True,
                                             zorder=ZOrder.OBS.value,
                                             **kwargs))
        # Contour behaves badly with warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.obs_contours.append(ax.contour(factor * self.obs_grid[..., 0],
                                                factor * self.obs_grid[..., 1],
                                                self.obs_total_values, [0]))

    def draw_pen(self):
        self.clear_pen()
        if self.penalty is None:
            return
        if self.coords == Coords.CARTESIAN:
            ax = self.main_ax
            kwargs = {}
            factor = 1.
        else:
            ax = self.ax
            kwargs = {'latlon': True}
            factor = Utils.RAD_TO_DEG
        matplotlib.rcParams['hatch.color'] = (.2, .2, .2, 1.)
        i, a = index(self.tcur, self.penalty['ts'])
        data = (1 - a) * self.penalty['data'][i] + a * self.penalty['data'][i + 1]
        self.pen_contours.append(ax.contourf(factor * self.penalty['grid'][..., 0],
                                             factor * self.penalty['grid'][..., 1],
                                             data,
                                             alpha=0.5,
                                             antialiased=True,
                                             zorder=ZOrder.OBS.value,
                                             **kwargs))

        if self.coords == Coords.CARTESIAN:
            self.pen_contours.append(ax.contour(factor * self.penalty['grid'][..., 0],
                                                factor * self.penalty['grid'][..., 1],
                                                data, [-0.5]))

    def draw_all(self):
        if self.mode_3d:
            self.draw_ff()
            self.draw_trajs()
        else:
            self.draw_ff()
            self.draw_trajs()
            if self.has_display_rff:
                self.draw_rff()
            self.draw_obs()
            self.draw_pen()
            self.draw_solver()
            if self.leg is None:
                self.leg = self.main_ax.legend(handles=self.leg_handles, labels=self.leg_labels, loc='center left',
                                               bbox_to_anchor=(1.2, 0.2), handletextpad=0.5, handlelength=0.5,
                                               markerscale=2)
            self.main_fig.canvas.draw()

    def draw_calibration(self):
        if self.coords == Coords.GCS:
            self.map.drawgreatcircle(Utils.RAD_TO_DEG * self.ff['grid'][:, 0, 0].min(),
                                     Utils.RAD_TO_DEG * self.ff['grid'][0, :, 1].min(),
                                     Utils.RAD_TO_DEG * self.ff['grid'][:, -1, 0].max(),
                                     Utils.RAD_TO_DEG * self.ff['grid'][-1, :, 1].max())
            self.map.drawgreatcircle(Utils.RAD_TO_DEG * self.ff['grid'][:, 0, 0].min(),
                                     Utils.RAD_TO_DEG * self.ff['grid'][0, :, 1].max(),
                                     Utils.RAD_TO_DEG * self.ff['grid'][:, -1, 0].max(),
                                     Utils.RAD_TO_DEG * self.ff['grid'][-1, :, 1].min())

    @property
    def img_fpath(self):
        return os.path.join(self.case_dir, f'thumbnail.{self.img_params["format"]}')

    def set_title(self, title):
        self.title = title

    def reload(self):
        # TODO: reconsider function
        t_start = time.time()
        print('Reloading... ', end='')

        # Reload params
        self.load_all()

        self.draw_all()

        t_end = time.time()
        print(f'Done ({t_end - t_start:.3f}s)')

    def switch_agg(self):
        if len(self.ef_agg_display) > 0:
            ef_id = 0
            for ef_id, b in self.ef_agg_display.items():
                if b:
                    break

            self.ef_agg_display[ef_id] = False
            self.ef_agg_display[(ef_id + 1) % len(self.ef_agg_display.keys())] = True

            self.draw_all()

    def toggle_agg(self):
        self.mode_aggregated = not self.mode_aggregated
        self.draw_all()

    def reload_time(self, val):
        self.tcur = self.tl + val * (self.tu - self.tl)
        noyear = False
        if self.tcur < 2000000:
            # Posix time less than 100 days after 1970-01-01 so
            # date does not refer to real time
            noyear = True
        try:
            d = datetime.fromtimestamp(self.tcur)
        except ValueError:
            d = None
        maj_time = str(d).split(".")[0]
        if noyear:
            maj_time = maj_time.split('-')[2]
        self.ax_timedisplay.set_text(f'{maj_time}')
        self.ax_timedisp_minor.set_text(f'{self.tcur:.3f}')

        self.draw_all()

    def legend(self):
        self.main_fig.legend()

    def toggle_controls(self):
        self.mode_controls = not self.mode_controls
        if not self.mode_controls:
            for a in self.traj_controls:
                a.remove()
            self.traj_controls = []
        self.draw_all()

    def toggle_rff(self):
        self.has_display_rff = not self.has_display_rff
        if not self.has_display_rff:
            self.clear_rff()
        self.draw_all()

    def toggle_ef(self):
        self.mode_ef = not self.mode_ef
        self.draw_all()

    def toggle_annot(self):
        self.mode_annot = not self.mode_annot
        self.draw_all()

    def toggle_ff_colors(self):
        self.mode_ff_color = not self.mode_ff_color
        if self.mode_ff_color:
            self.selected_cm = custom_cm
        else:
            self.selected_cm = custom_desat_cm
        self.draw_all()

    def toggle_energy(self):
        self.mode_energy = not self.mode_energy
        self.draw_all()

    def toggle_ef_display(self):
        self.mode_ef_display = not self.mode_ef_display
        self.draw_all()

    def increment_time(self, k=1):
        val = self.tcur / (self.tu - self.tl)
        next_val = val + self.increment_factor * k
        if 0. <= next_val <= 1.:
            self.time_slider.set_val(val)
            self.reload_time(next_val)

    def keyboard(self, event):
        if event.key == 's':
            self.switch_agg()
        elif event.key == 'r':
            self.reload()
        elif event.key == 'f':
            self.toggle_agg()
        elif event.key == 't':
            self.toggle_rff()
        elif event.key == 'c':
            self.toggle_controls()
        elif event.key == 'h':
            self.toggle_ef()
        elif event.key == 'right':
            self.increment_time()
        elif event.key == 'left':
            self.increment_time(k=-1)
        elif event.key == 'a':
            self.toggle_annot()
        elif event.key == 'l':
            self.toggle_ff_colors()
        elif event.key == 'e':
            self.toggle_energy()
        elif event.key == 'x':
            self.toggle_ef_display()

    def show(self, block=True):
        self.main_fig.savefig(self.img_fpath, **self.img_params)

        self.main_fig.canvas.mpl_connect('key_press_event', self.keyboard)
        plt.show(block=block)

    def set_mode(self, flags):
        if flags is None:
            return
        if 's' in flags:
            self.switch_agg()
        if 'a' in flags:
            self.mode_aggregated = True
        if 't' in flags:
            self.toggle_rff()
        if 'h' in flags:
            self.mode_ef = False
        if 'u' in flags:
            self.rescale_ff = False
        if 'e' not in flags:
            self.mode_energy = False

    def to_movie(self, frames=50, fps=10, movie_format='mp4', mini=False):
        self._info('Rendering animation')
        anim_path = os.path.join(self.case_dir, 'anim')
        if not os.path.exists(anim_path):
            os.mkdir(anim_path)
        else:
            for filename in os.listdir(anim_path):
                os.remove(os.path.join(anim_path, filename))
        r = range(frames)
        try:
            from tqdm import tqdm
            r = tqdm(r)
        except ImportError:
            warnings.warn('"tqdm" package is not installed for progress bar')
        for i in r:
            val = i / (frames - 1)
            self.time_slider.set_val(val)
            self.reload_time(val)
            kwargs = {}
            if mini:
                extent = self.main_ax.get_window_extent().transformed(self.main_fig.dpi_scale_trans.inverted())
                kwargs['bbox_inches'] = extent
                kwargs['dpi'] = 50
            self.main_fig.savefig(os.path.join(anim_path, f'{i:0>4}.png'), **kwargs)

        pattern_in = os.path.join(anim_path, '*.png')
        first_file_in = os.path.join(anim_path, '0000.png')
        palette = os.path.join(self.case_dir, 'palette.png')

        if movie_format == 'raw':
            return
        elif movie_format == 'gif':
            os.system(f"ffmpeg -i {first_file_in} -y -vf palettegen {palette}")
            file_out = os.path.join(self.case_dir, f'anim_s.{movie_format}')
            command = f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{pattern_in}' -i {palette} -filter_complex 'paletteuse' '{file_out}'"
            os.system(command)
            os.remove(palette)
        else:
            file_out = os.path.join(self.case_dir, f'anim.{movie_format}')
            command = f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{pattern_in}' '{file_out}'"
            # -c: v libx264 - pix_fmt yuv420p
            os.system(command)
        for filename in os.listdir(anim_path):
            if filename.endswith('.png'):
                os.remove(os.path.join(anim_path, filename))
        os.rmdir(anim_path)

    @staticmethod
    def _info(msg):
        print(f'[display] {msg}')

    @property
    def coords(self):
        return self.io.coords


def index(t, t_list) -> tuple[int, float]:
    """
    Get nearest lowest index for time discrete grid and interpolation coefficient.
    If some data f is known on the increasing grid t_list, then its interpolated value at t is
    (1 - alpha) * f[index] + alpha * f[index + 1]
    :param t: Time for interpolation
    :param t_list: Time grid
    :return: index: Nearest lowest index for time, alpha: coefficient for the linear interpolation
    """
    nt = t_list.shape[0]
    if t <= t_list[0]:
        return 0, 0.
    if t > t_list[-1]:
        # Freeze ff to last frame
        return nt - 2, 1.
    tau = (t - t_list[0]) / (t_list[-1] - t_list[0])
    i, alpha = int(tau * (nt - 1)), tau * (nt - 1) - int(tau * (nt - 1))
    if i == nt - 1:
        i = nt - 2
        alpha = 1.
    return i, alpha
