import json
import os
import sys
import time
import warnings
import webbrowser
from datetime import datetime
from math import floor
from typing import Optional

import h5py
import matplotlib
import matplotlib.cm as mpl_cm
import scipy.ndimage
import tqdm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from mpl_toolkits.basemap import Basemap
from numpy import ndarray
from pyproj import Proj, Geod
from scipy.interpolate import griddata

from misc import *

state_names = [r"$x\:[m]$", r"$y\:[m]$"]
control_names = [r"$u\:[rad]$"]


class Display:
    """
    Defines all the visualization functions for navigation problems
    """

    def __init__(self, coords='cartesian', title='Title', mode_3d=False):
        """
        :param coords: Either 'cartesian' for planar problems or 'gcs' for Earth-based problems
        """
        self.coords = coords

        self.x_min: Optional[float] = None
        self.x_max: Optional[float] = None
        self.y_min: Optional[float] = None
        self.y_max: Optional[float] = None

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
        self.airspeed = None
        self.title = title
        self.axes_equal = True
        self.projection = 'ortho'
        self.sm_wind = None
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

        self.output_dir = None
        self.output_imgpath = None
        self.img_params = {
            'dpi': 300,
            'format': 'png'
        }
        self.params_fname = 'pb_info.json'
        self.params_fname_formatted = 'params.html'

        self.wind_fname = 'wind.h5'
        self.trajs_fname = 'trajectories.h5'
        self.rff_fname = 'rff.h5'
        self.obs_fname = 'obs.h5'
        self.pen_fname = 'penalty.h5'
        self.filter_fname = '.trajfilter'
        self.wind_fpath = None
        self.trajs_fpath = None
        self.rff_fpath = None
        self.obs_fpath = None
        self.pen_fpath = None
        self.filter_fpath = None

        self.traj_filter = []
        # 0 for no aggregation (fronts), 1 for aggregation and time cursor, 2 for aggrgation and no time cursor
        self.mode_aggregated = 0
        self.mode_controls = False  # Whether to print wind in tiles (False) or by interpolation (True)
        self.mode_wind = False  # Whether to display extremal field or not
        self.mode_ef = True  # Whether to display zones where windfield is equal to airspeed
        self.mode_speed = True  # Whether to display trajectories annotation
        self.mode_annot = False  # Whether to display wind colors
        self.mode_wind_color = True  # Whether to display energy as colors
        self.mode_energy = True  # Whether to draw extremal fields or not
        self.mode_ef_display = True  # Whether to rescale wind
        self.rescale_wind = True

        # True if wind norm colobar is displayed, False if energy colorbar is displayed
        self.active_windcb = True

        self.has_display_rff = True

        self.wind = None
        self.wind_norm_min = None
        self.wind_norm_max = None
        self.trajs = []
        self.rff = None
        self.rff_cntr_kwargs = {
            'zorder': ZO_RFF,
            'cmap': 'brg',
            'antialiased': True,
            'alpha': .8,
        }
        self.rff_zero_ceils = None
        self.nx_rft = None
        self.ny_rft = None
        self.nt_rft = None

        self.obstacles = None
        self.obs_grid = None

        self.penalty = None

        self.tv_wind = False

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

        self.label_list = []
        self.scatter_init = None
        self.scatter_target = None

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

        self.wind_colormesh = None
        self.wind_colorcontour = None
        self.wind_ceil = None
        self.wind_quiver = None
        self.ax_timeslider = None
        self.time_slider = None
        self.wind_colorbar = None
        self.ax_timedisplay = None
        self.ax_timedisp_minor = None
        # Time window lower bound
        # Defined as wind time lower bound if wind is time-varying
        # Else minimum timestamps among trajectories and fronts
        self.tl = None
        # Time window upper bound
        # Maximum among all upper bounds (wind, trajs, rffs)
        self.tu = None

        self.tl_wind = None
        self.tu_wind = None

        self.tl_traj = None
        self.tu_traj = None

        self.tl_rff = None
        self.tu_rff = None

        # Current time step to display
        self.tcur = None

        self.color_mode = ''

        self.ef_nt = 1000
        self.ef_ids = []
        self.ef_index = None
        self.ef_trajgroups = {}
        self.ef_fronts = {}
        self.ef_ts = None
        # True or False dict whether group of extremals should be
        # displayed in aggregated mode
        self.ef_agg_display = {}

        self.increment_factor = 0.001

        self.x_init = None
        self.x_target = None

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        # matplotlib.rcParams['text.usetex'] = True

        self.mode_3d = mode_3d

    def _index(self, mode):
        """
        Get nearest lowest index for time discrete grid
        :param mode: 'wind', 'rff' or 'pen'
        :return: Nearest lowest index for time, coefficient for the linear interpolation
        """
        ts = None
        if mode == 'wind':
            ts = self.wind['ts']
        elif mode == 'rff':
            ts = self.rff['ts']
        elif mode == 'pen':
            ts = self.penalty['ts']
        else:
            Display._error(f'Unknown mode "{mode}" for _index')
        nt = ts.shape[0]
        if self.tcur <= ts[0]:
            return 0, 0.
        if self.tcur > ts[-1]:
            # Freeze wind to last frame
            return nt - 2, 1.
        tau = (self.tcur - ts[0]) / (ts[-1] - ts[0])
        i, alpha = int(tau * (nt - 1)), tau * (nt - 1) - int(tau * (nt - 1))
        if i == nt - 1:
            i = nt - 2
            alpha = 1.
        return i, alpha

    def setup(self, bl_off=None, tr_off=None, projection='ortho', debug=False):
        self.projection = projection

        if self.bl_man is not None:
            self.has_manual_bbox = True
            if type(self.bl_man) == str:
                raise Exception('Name translation unsupported')
            else:
                self.x_min = self.bl_man[0]
                self.y_min = self.bl_man[1]

            if type(self.tr_man) == str:
                raise Exception('Name translation unsupported')
            else:
                self.x_max = self.tr_man[0]
                self.y_max = self.tr_man[1]
        else:
            self.set_wind_bounds()

        if bl_off is not None:
            self.x_offset = bl_off

        if tr_off is not None:
            self.y_offset = tr_off

        # if self.opti_ceil is None:
        #     self.opti_ceil = 0.0005 * 0.5 * (self.x_max - self.x_min + self.y_max - self.y_min)

        if debug:
            print(self.x_min, self.x_max, self.y_min, self.y_max)

        self.fsc = FontsizeConf()
        plt.rc('font', size=self.fsc.fontsize)
        plt.rc('axes', titlesize=self.fsc.axes_titlesize)
        plt.rc('axes', labelsize=self.fsc.axes_labelsize)
        plt.rc('xtick', labelsize=self.fsc.xtick_labelsize)
        plt.rc('ytick', labelsize=self.fsc.ytick_labelsize)
        plt.rc('legend', fontsize=self.fsc.legend_fontsize)
        # plt.rc('font', family=self.fsc.font_family)
        plt.rc('mathtext', fontset=self.fsc.mathtext_fontset)

        self.main_fig = plt.figure(num=f"dabryvisu ({self.coords})",
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
        cartesian = (self.coords == 'cartesian')
        gcs = (self.coords == 'gcs')

        if self.mode_3d:
            self.ax = self.main_ax
            self.ax.set_xlim(self.x_min - self.x_offset * (self.x_max - self.x_min),
                                 self.x_max + self.x_offset * (self.x_max - self.x_min))
            self.ax.set_ylim(self.y_min - self.y_offset * (self.y_max - self.y_min),
                                 self.y_max + self.y_offset * (self.y_max - self.y_min))
            self.ax.set_zlim(0, self.engy_max)
            return

        if gcs:
            kwargs = {
                'resolution': 'c',
                'projection': self.projection,
                'ax': self.main_ax
            }
            # Don't plot coastal lines features less than 1000km^2
            # kwargs['area_thresh'] = (6400e3 * np.pi / 180) ** 2 * (self.x_max - self.x_min) * 0.5 * (
            #             self.y_min + self.y_max) / 1000.

            if self.projection == 'merc':
                kwargs['llcrnrlon'] = RAD_TO_DEG * (self.x_min - self.x_offset * (self.x_max - self.x_min))
                kwargs['llcrnrlat'] = RAD_TO_DEG * (self.y_min - self.y_offset * (self.y_max - self.y_min))
                kwargs['urcrnrlon'] = RAD_TO_DEG * (self.x_max + self.x_offset * (self.x_max - self.x_min))
                kwargs['urcrnrlat'] = RAD_TO_DEG * (self.y_max + self.y_offset * (self.y_max - self.y_min))

            elif self.projection == 'ortho':
                kwargs['lon_0'], kwargs['lat_0'] = \
                    0.5 * RAD_TO_DEG * (np.array((self.x_min, self.y_min)) + np.array((self.x_max, self.y_max)))
                # tuple(RAD_TO_DEG * np.array(middle(np.array((self.x_min, self.y_min)),
                #                                    np.array((self.x_max, self.y_max)))))
                proj = Proj(proj='ortho', lon_0=kwargs['lon_0'], lat_0=kwargs['lat_0'])
                # pgrid = np.array(proj(RAD_TO_DEG * self.wind['grid'][:, :, 0],
                # RAD_TO_DEG * self.wind['grid'][:, :, 1]))
                # px_min = np.min(pgrid[0])
                # px_max = np.max(pgrid[0])
                # py_min = np.min(pgrid[1])
                # py_max = np.max(pgrid[1])
                if self.x_init is not None and self.x_target is not None:
                    x1, y1 = proj(*(RAD_TO_DEG * self.x_init))
                    x2, y2 = proj(*(RAD_TO_DEG * self.x_target))
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
                kwargs['lon_1'] = RAD_TO_DEG * self.x_init[0]
                kwargs['lat_1'] = RAD_TO_DEG * self.x_init[1]
                kwargs['lon_2'] = RAD_TO_DEG * self.x_target[0]
                kwargs['lat_2'] = RAD_TO_DEG * self.x_target[1]
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
                kwargs['lon_0'], kwargs['lat_0'] = RAD_TO_DEG * lon_0, RAD_TO_DEG * lat_0
                kwargs['width'] = distance(self.x_init, self.x_target) * 1.35
                kwargs['height'] = distance(self.x_init, self.x_target) * 1.35

            elif self.projection == 'lcc':
                kwargs['lon_0'] = RAD_TO_DEG * 0.5 * (self.x_min + self.x_max)
                kwargs['lat_0'] = RAD_TO_DEG * 0.5 * (self.y_min + self.y_max)
                kwargs['lat_1'] = RAD_TO_DEG * (self.y_max + self.y_offset * (self.y_max - self.y_min))
                kwargs['lat_2'] = RAD_TO_DEG * (self.y_min - self.y_offset * (self.y_max - self.y_min))
                kwargs['width'] = (1 + self.x_offset) * (self.x_max - self.x_min) * EARTH_RADIUS
                kwargs['height'] = kwargs['width']

            else:
                print(f'Projection type "{self.projection}" not handled yet', file=sys.stderr)
                exit(1)

            self.map = Basemap(**kwargs)

            # self.map.shadedrelief()
            self.map.drawcoastlines()
            self.map.fillcontinents()
            if self.x_init is not None and self.x_target is not None:
                self.map.drawgreatcircle(RAD_TO_DEG * self.x_init[0],
                                  RAD_TO_DEG * self.x_init[1],
                                  RAD_TO_DEG * self.x_target[0],
                                  RAD_TO_DEG * self.x_target[1], color='grey', linestyle='--')
            lw = 0.5
            dashes = (2, 2)
            # draw parallels
            try:
                lat_min = RAD_TO_DEG * min(self.y_min, self.y_max)
                lat_max = RAD_TO_DEG * max(self.y_min, self.y_max)
                n_lat = floor((lat_max - lat_min) / 10) + 2
                self.map.drawparallels(10. * (floor(lat_min / 10.) + np.arange(n_lat)),  # labels=[1, 0, 0, 0],
                                       linewidth=lw,
                                       dashes=dashes, textcolor=(1., 1., 1., 0.))
                # draw meridians
                lon_min = RAD_TO_DEG * min(self.x_min, self.x_max)
                lon_max = RAD_TO_DEG * max(self.x_min, self.x_max)
                n_lon = floor((lon_max - lon_min) / 10) + 2
                self.map.drawmeridians(10. * (floor(lon_min / 10.) + np.arange(n_lon)),  # labels=[1, 0, 0, 1],
                                       linewidth=lw,
                                       dashes=dashes)
            except ValueError:
                pass

        if cartesian:
            if self.axes_equal:
                self.main_ax.axis('equal')
            self.main_ax.set_xlim(self.x_min - self.x_offset * (self.x_max - self.x_min),
                                  self.x_max + self.x_offset * (self.x_max - self.x_min))
            self.main_ax.set_ylim(self.y_min - self.y_offset * (self.y_max - self.y_min),
                                  self.y_max + self.y_offset * (self.y_max - self.y_min))
            self.main_ax.set_xlabel('$x$ [m]')
            self.main_ax.set_ylabel('$y$ [m]')
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

    def set_wind_bounds(self):
        self.x_min = np.min(self.wind['grid'][:, :, 0])
        self.x_max = np.max(self.wind['grid'][:, :, 0])
        self.y_min = np.min(self.wind['grid'][:, :, 1])
        self.y_max = np.max(self.wind['grid'][:, :, 1])

    def draw_point_by_name(self, name):
        raise Exception('Name translation unsupported')
        # if self.coords != 'gcs':
        #     print('"draw_point_by_name" only available for GCS coordinates')
        #     exit(1)
        # self.map.scatter(loc[0], loc[1], s=8., color='red', marker='D', latlon=True, zorder=ZO_ANNOT)
        # self.mainax.annotate(name, self.map(loc[0], loc[1]), (10, 10), textcoords='offset pixels', ha='center')

    def draw_point(self, x, y, label=None):
        kwargs = {
            's': 8.,
            'color': 'red',
            'marker': 'D',
            'zorder': ZO_ANNOT
        }
        if self.coords == 'gcs':
            kwargs['latlon'] = True
            pos_annot = self.map(x, y)
        else:
            pos_annot = (x, y)
        self.ax.scatter(x, y, **kwargs)
        if label is not None:
            self.ax.annotate(label, pos_annot, (10, 10), textcoords='offset pixels', ha='center')

    def clear_wind(self):
        if self.wind_colormesh is not None:
            self.wind_colormesh.remove()
            self.wind_colormesh = None
        if self.wind_colorcontour is not None:
            try:
                self.wind_colorcontour.remove()
            except AttributeError:
                for coll in self.wind_colorcontour.collections:
                    coll.remove()
            self.wind_colorcontour = None
        if self.wind_ceil is not None:
            for coll in self.wind_ceil.collections:
                coll.remove()
            self.wind_ceil = None
        if self.wind_quiver is not None:
            self.wind_quiver.remove()
            self.wind_quiver = None
        # if self.wind_colorbar is not None:
        #     self.wind_colorbar.remove()
        #     self.wind_colorbar = None

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

    def load_filter(self):
        self.filter_fpath = os.path.join(self.output_dir, self.filter_fname)
        if not os.path.exists(self.filter_fpath):
            return
        with open(self.filter_fpath, 'r') as f:
            self.traj_filter = list(map(str.strip, f.readlines()))

    def load_wind(self, filename=None):
        self.wind = None
        if self.wind_fpath is None:
            filename = self.wind_fname if filename is None else filename
            self.wind_fpath = os.path.join(self.output_dir, filename)
        with h5py.File(self.wind_fpath, 'r') as f:
            self.wind = {}
            self.wind['data'] = np.zeros(f['data'].shape)
            self.wind['data'][:] = f['data']
            norms = np.linalg.norm(f['data'], axis=3)
            self.wind_norm_min, self.wind_norm_max = np.min(norms), np.max(norms)
            self.wind['attrs'] = {}
            for k, v in f.attrs.items():
                self.wind['attrs'][k] = v
            self.wind['grid'] = np.zeros(f['grid'].shape)
            self.wind['grid'][:] = f['grid']
            self.wind['ts'] = np.zeros(f['ts'].shape)
            self.wind['ts'][:] = f['ts']
            if self.wind['ts'].shape[0] > 1:
                self.tv_wind = True
                self.tl_wind = self.wind['ts'][0]
                self.tu_wind = self.wind['ts'][-1]

    def load_trajs(self, filename=None):
        self.trajs = []
        if self.trajs_fpath is None:
            filename = self.trajs_fname if filename is None else filename
            trajfiles = [name for name in os.listdir(self.output_dir)
                         if name.startswith(self.trajs_fname.split('.')[0])]
            self.trajs_fpath = list(map(lambda fn: os.path.join(self.output_dir, fn), trajfiles))
        if len(self.trajs_fpath) == 0 or not os.path.exists(self.trajs_fpath[0]):
            return
        for traj_fpath in self.trajs_fpath:
            with h5py.File(traj_fpath, 'r') as f:
                for k, traj in enumerate(f.values()):
                    if 'info' in traj.attrs.keys() and traj.attrs['info'] in self.traj_filter:
                        continue
                    if traj['ts'].shape[0] <= 1:
                        continue
                    if traj.attrs['coords'] != self.coords:
                        print(
                            f'[Warning] Traj. coord type {traj.attrs["coords"]} differs from display mode {self.coords}')

                    """
                    if self.nt_tick is not None:
                        kwargs['nt_tick'] = self.nt_tick
                    """
                    _traj = {}
                    _traj['data'] = np.zeros(traj['data'].shape)
                    _traj['data'][:] = traj['data']
                    #_traj['controls'] = np.zeros(traj['controls'].shape)
                    #_traj['controls'][:] = traj['controls']
                    _traj['ts'] = np.zeros(traj['ts'].shape)
                    _traj['ts'][:] = traj['ts']
                    if _traj['ts'].shape[0] == 0:
                        continue

                    # if 'airspeed' in traj.keys():
                    #     _traj['airspeed'] = np.zeros(traj['airspeed'].shape)
                    #     _traj['airspeed'][:] = traj['airspeed']

                    if 'energy' in traj.keys() and traj['energy'].shape[0] > 0:
                        _traj['energy'] = np.zeros(traj['energy'].shape)
                        _traj['energy'][:] = traj['energy']
                        cmin = _traj['energy'][:].min()
                        cmax = _traj['energy'][:].max()
                        if self.engy_min is None or cmin < self.engy_min:
                            self.engy_min = cmin
                        if self.engy_max is None or cmax > self.engy_max:
                            self.engy_max = cmax

                    # Backward compatibility
                    if 'info' in traj.attrs.keys():
                        _traj['info'] = traj.attrs['info']
                    else:
                        _traj['info'] = ''

                    if 'info_dict' in traj.attrs.keys():
                        _traj['info_dict'] = traj['info_dict']

                    # Label trajectories belonging to extremal fields
                    if _traj['info'].startswith('ef'):
                        ef_id = int(_traj['info'].strip().split('_')[1])
                        if ef_id not in self.ef_ids:
                            self.ef_ids.append(ef_id)
                            self.ef_trajgroups[ef_id] = []
                        self.ef_trajgroups[ef_id].append(_traj)

                    # Adapt time window to trajectories' timestamps

                    tl = np.min(_traj['ts'])
                    if self.tl_traj is None or tl < self.tl_traj:
                        self.tl_traj = tl
                    tu = np.max(_traj['ts'])
                    if self.tu_traj is None or tu > self.tu_traj:
                        self.tu_traj = tu

                    self.trajs.append(_traj)
        # self.engy_min = 0.
        # self.engy_max = 16 * 3.6e6
        self.engy_norm = mpl_colors.Normalize(vmin=self.engy_min, vmax=self.engy_max)

    def load_rff(self, filename=None):
        self.rff = None
        self.rff_zero_ceils = []
        if self.rff_fpath is None:
            filename = self.rff_fname if filename is None else filename
            self.rff_fpath = os.path.join(self.output_dir, filename)
        if not os.path.exists(self.rff_fpath):
            return
        with h5py.File(self.rff_fpath, 'r') as f:
            if self.coords != f.attrs['coords']:
                Display._warn(f'RFF coords "{f.attrs["coords"]}" does not match current display coords "{self.coords}"')

            if self.coords == 'gcs':
                self.rff_cntr_kwargs['latlon'] = True
            self.nt_rft, self.nx_rft, self.ny_rft = f['data'].shape
            # ceil = min((self.x_max - self.x_min) / (3 * self.nx_rft),
            #            (self.y_max - self.y_min) / (3 * self.ny_rft))
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
                Display._warn(f'No RFF value in zero band for indexes {tuple(failed_zeros)}')

    def load_obs(self, filename=None):
        if self.obs_fpath is None:
            filename = self.obs_fname if filename is None else filename
            self.obs_fpath = os.path.join(self.output_dir, filename)
        if not os.path.exists(self.obs_fpath):
            return
        with h5py.File(self.obs_fpath, 'r') as f:
            self.obstacles = np.zeros(f['data'].shape)
            self.obstacles[:] = f['data']
            self.obs_grid = np.zeros(f['grid'].shape)
            self.obs_grid[:] = f['grid']

    def load_pen(self, filename=None):
        if self.pen_fpath is None:
            filename = self.pen_fname if filename is None else filename
            self.pen_fpath = os.path.join(self.output_dir, filename)
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
        self.load_wind()
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
        n_ef = len(self.ef_trajgroups.keys())
        n_eft = sum((len(tgv) for tgv in self.ef_trajgroups.values()))
        Display._info(
            f'Loading completed. {n_trajs - n_eft} regular trajs, {n_ef} extremal fields of {n_eft} trajs, {n_rffs} RFFs.')

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
            self.tl = self.tl_wind
            self.tu = self.tu_wind
        else:
            self.tl = mysat(self.tl_traj, self.tl_rff, minimize=True)
            self.tu = mysat(self.tu_traj, self.tu_rff, minimize=False)

    def import_params(self, fname=None):
        """
        Load necessary data from parameter file.
        Currently only loads time window upper bound and trajectory ticking option
        :param fname: The parameters filename if different from standard
        """
        fname = self.params_fname if fname is None else fname
        fpath = os.path.join(self.output_dir, fname)
        success = False
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                self.params = json.load(f)
            try:
                self.coords = self.params['coords']
                success = True
            except KeyError:
                pass
            try:
                self.bl_man = np.array(self.params['bl_pb'])
                success = True
            except KeyError:
                pass
            try:
                self.tr_man = np.array(self.params['tr_pb'])
                success = True
            except KeyError:
                pass
        if not success:
            # Try to recover coords from present data
            noted_coords = set()
            for ffname in os.listdir(self.output_dir):
                if ffname.endswith('.h5'):
                    f = h5py.File(os.path.join(self.output_dir, ffname), 'r')
                    try:
                        noted_coords.add(f.attrs['coords'])
                    except KeyError:
                        # May be groups of trajectories
                        try:
                            noted_coords.add(f['0'].attrs['coords'])
                        except KeyError:
                            pass
            if len(noted_coords) == 0:
                Display._error(f'Cannot establish coords type for case "{os.path.join(self.output_dir), fname}"')
            elif len(noted_coords) >= 2:
                Display._error(
                    f'Coordinates type inconsistent for problem data : "{os.path.join(self.output_dir), fname}"')
            else:
                self.coords = noted_coords.pop()

        missing = set(self.p_names).difference(self.params.keys())
        try:
            self.x_init = np.array(self.params['x_init'])
            missing.remove(f'x_init')
        except KeyError:
            # Backward compatibility
            try:
                self.x_init = np.array(self.params['point_init'])
                missing.remove(f'x_init')
            except KeyError:
                pass
        try:
            self.x_target = np.array(self.params['x_target'])
            missing.remove(f'x_target')
        except KeyError:
            try:
                self.x_target = np.array(self.params['point_target'])
                missing.remove(f'x_target')
            except KeyError:
                pass

        for name in ['airspeed', 'va', 'v_a']:
            if name in self.params.keys():
                self.airspeed = self.params[name]

        # if self.airspeed is not None:
        #     self.cm_norm_min = 0.
        #     self.cm_norm_max = 2 * self.airspeed

        if len(missing) > 0:
            Display._info(f'Missing parameters : {tuple(missing)}')

    def preprocessing(self):
        # Preprocessing
        # For extremal fields, group points in fronts
        no_display = False
        for ef_id in self.ef_ids:
            self.ef_ts = np.linspace(self.tl, self.tu, self.ef_nt)
            front = [[] for _ in range(self.ef_nt)]

            self.ef_index = lambda t: None if t < self.tl or t > self.tu else \
                floor((self.ef_nt - 1) * (t - self.tl) / (self.tu - self.tl))

            for traj in self.ef_trajgroups[ef_id]:
                nt = len(traj['ts'])
                for k, t in enumerate(traj['ts']):
                    i = self.ef_index(t)
                    if i is not None:
                        ke = k if k < nt - 1 else k-1
                        try:
                            e = traj['energy'][ke]
                        except KeyError:
                            e = 0.
                        front[i].append(np.array(tuple(traj['data'][k]) + (e,)))
            for i in range(len(front)):
                front[i] = np.array(front[i])
            self.ef_fronts[ef_id] = front
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
                    factor = RAD_TO_DEG if self.coords == 'gcs' else 1.
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
                i, alpha = self._index('rff')
                rff_values = (1 - alpha) * self.rff['data'][i, :, :] + alpha * self.rff['data'][i + 1, :, :]
                zero_ceil = (1 - alpha) * self.rff_zero_ceils[i] + alpha * self.rff_zero_ceils[i + 1]
                if self.coords == 'gcs':
                    nx, ny = self.rff['grid'][:, :, 0].shape
                    xi = np.linspace(self.rff['grid'][:, :, 0].min(), self.rff['grid'][:, :, 0].max(), 4 * nx)
                    yi = np.linspace(self.rff['grid'][:, :, 1].min(), self.rff['grid'][:, :, 1].max(), 4 * ny)
                    xi, yi = np.meshgrid(xi, yi)

                    # interpolate
                    x, y, z = self.rff['grid'][:, :, 0], self.rff['grid'][:, :, 1], rff_values
                    zi = griddata((x.reshape(-1), y.reshape(-1)), z.reshape(-1), (xi, yi))
                    args = (RAD_TO_DEG * xi, RAD_TO_DEG * yi, zi)
                else:
                    args = (self.rff['grid'][:, :, 0], self.rff['grid'][:, :, 1], rff_values)
                args = args + ([-zero_ceil / 2, zero_ceil / 2],)
                # ([-1000., 1000.],) if not debug else (np.linspace(-100000, 100000, 200),))
                self.rff_contours.append(self.ax.contourf(*args, **self.rff_cntr_kwargs))

    def draw_wind(self, showanchors=False):

        # Erase previous drawings if existing
        self.clear_wind()

        nt, nx, ny, _ = self.wind['data'].shape
        X = np.zeros((nx, ny))
        Y = np.zeros((nx, ny))
        if self.rescale_wind:
            ur = max(1, nx // 18)  # nx // 20
        else:
            ur = 1
        factor = RAD_TO_DEG if self.coords == 'gcs' else 1.
        X[:] = factor * self.wind['grid'][:, :, 0]
        Y[:] = factor * self.wind['grid'][:, :, 1]

        alpha_bg = 0.7

        U_grid = np.zeros((nx, ny))
        V_grid = np.zeros((nx, ny))
        if not self.tv_wind:
            U_grid[:] = self.wind['data'][0, :, :, 0]
            V_grid[:] = self.wind['data'][0, :, :, 1]
        else:
            k, p = self._index('wind')
            U_grid[:] = (1 - p) * self.wind['data'][k, :, :, 0] + p * self.wind['data'][k + 1, :, :, 0]
            V_grid[:] = (1 - p) * self.wind['data'][k, :, :, 1] + p * self.wind['data'][k + 1, :, :, 1]
        U = U_grid.flatten()
        V = V_grid.flatten()

        norms3d = np.sqrt(U_grid ** 2 + V_grid ** 2)
        norms = np.sqrt(U ** 2 + V ** 2)
        eps = (np.max(norms) - np.min(norms)) * 1e-6
        # norms += eps

        norm = mpl_colors.Normalize()
        self.cm = self.selected_cm
        if self.coords == 'gcs' and self.wind_norm_max < 1.5 * self.cm_norm_max:
            self.cm = windy_cm
            norm.autoscale(np.array([self.cm_norm_min, self.cm_norm_max]))
        else:
            self.cm = jet_desat_cm
            norm.autoscale(np.array([self.wind_norm_min, self.wind_norm_max]))

        needs_engy = self.mode_energy and self.mode_ef and self.mode_aggregated
        set_engycb = needs_engy and self.active_windcb
        if self.sm_wind is None:
            self.sm_wind = mpl_cm.ScalarMappable(cmap=self.cm, norm=norm)
            if self.engy_min and self.engy_max is not None:
                self.sm_engy = mpl_cm.ScalarMappable(cmap='tab20b',
                                                     norm=mpl_colors.Normalize(
                                                         vmin=self.engy_min / 3.6e6,
                                                         vmax=self.engy_max / 3.6e6))
            if self.coords == 'gcs':
                self.wind_colorbar = self.ax.colorbar(self.sm_wind, pad=0.1)
            elif self.coords == 'cartesian':
                self.wind_colorbar = self.main_fig.colorbar(self.sm_wind, ax=self.ax, pad=0.03)
            self.wind_colorbar.set_label('Wind [m/s]', labelpad=10)
            self.active_windcb = True

        set_windcb = not needs_engy and not self.active_windcb
        if set_windcb:
            self.wind_colorbar.update_normal(self.sm_wind)
            self.wind_colorbar.set_label('Wind [m/s]')
            self.active_windcb = True
        elif set_engycb and self.sm_engy is not None:
            self.wind_colorbar.update_normal(self.sm_engy)
            self.wind_colorbar.set_label('Energy [kWh]')
            self.active_windcb = False

        # Wind norm plot
        # znorms3d = scipy.ndimage.zoom(norms3d, 3)
        # zX = scipy.ndimage.zoom(X, 3)
        # zY = scipy.ndimage.zoom(Y, 3)
        znorms3d = norms3d
        zX = X
        zY = Y
        kwargs = {
            'cmap': self.cm,
            'norm': norm,
            'alpha': alpha_bg,
            'zorder': ZO_WIND_NORM,
        }
        if self.coords == 'gcs':
            kwargs['latlon'] = True

        if not self.mode_wind:
            if self.mode_3d:
                colors = self.cm(norm(znorms3d))
                self.wind_colormesh = self.ax.plot_surface(zX, zY, np.zeros(zX.shape), facecolors=colors, zorder=ZO_WIND_NORM)#, **kwargs)
            else:
                kwargs['shading'] = 'auto'
                kwargs['antialiased'] = True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.wind_colormesh = self.ax.pcolormesh(zX, zY, znorms3d, **kwargs)
        else:
            znorms3d = scipy.ndimage.zoom(norms3d, 3)
            zX = scipy.ndimage.zoom(X, 3)
            zY = scipy.ndimage.zoom(Y, 3)
            kwargs['antialiased'] = True
            kwargs['levels'] = 30
            self.wind_colorcontour = self.ax.contourf(zX, zY, znorms3d, **kwargs)

        # Wind ceil
        if self.airspeed is not None and self.mode_speed:
            if 'shading' in kwargs.keys():
                del kwargs['shading']
            # znorms3d = scipy.ndimage.zoom(norms3d, 3)
            # zX = scipy.ndimage.zoom(X, 3)
            # zY = scipy.ndimage.zoom(Y, 3)
            kwargs['antialiased'] = True
            ceil = (self.cm_norm_max - self.cm_norm_min) / 100.
            kwargs['levels'] = (self.airspeed - ceil, self.airspeed + ceil)
            self.wind_ceil = self.ax.contourf(X, Y, norms3d, **kwargs)

        # Quiver plot
        qX = X[::ur, ::ur]
        qY = Y[::ur, ::ur]
        qU = U_grid[::ur, ::ur].flatten()
        qV = V_grid[::ur, ::ur].flatten()
        qnorms = 1e-6 + np.sqrt(qU ** 2 + qV ** 2)
        kwargs = {
            'color': (0.2, 0.2, 0.2, 1.0),
            # 'width': 1,  # 0.001,
            'pivot': 'tail',
            'alpha': 0.7,
            'units': 'xy',
            'minshaft': 2.,
            'zorder': ZO_WIND_VECTORS
        }
        if self.coords == 'gcs':
            kwargs['latlon'] = True
        if np.any(qnorms > self.cm_norm_min + 0.01 * (self.cm_norm_max - self.cm_norm_min)):
            if not self.mode_3d:
                self.wind_quiver = self.ax.quiver(qX, qY, qU, qV, **kwargs)  # color=color)
            else:
                U, V = U_grid[::ur, ::ur], V_grid[::ur, ::ur]
                qZ = np.zeros(qX.shape)
                W = np.zeros(U.shape)
                del kwargs['units']
                del kwargs['minshaft']
                self.wind_quiver = self.ax.quiver(qX, qY, qZ, U, V, W, **kwargs)  # color=color)


    # def setup_components(self):
    #     for k, state_plot in enumerate(self.state):
    #         state_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
    #         state_plot.tick_params(direction='in')
    #         state_plot.yaxis.set_label_position("right")
    #         state_plot.yaxis.tick_right()
    #         state_plot.set_ylabel(state_names[k])
    #         plt.setp(state_plot.get_xticklabels(), visible=False)
    #
    #     for k, control_plot in enumerate(self.control):
    #         control_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
    #         control_plot.tick_params(direction='in')
    #         control_plot.yaxis.set_label_position("right")
    #         control_plot.yaxis.tick_right()
    #         control_plot.set_ylabel(control_names[k])
    #         # Last plot
    #         if k == len(self.control) - 1:
    #             control_plot.set_xlabel(r"$t\:[s]$")

    def draw_trajs(self, nolabels=False, opti_only=False):
        self.clear_trajs()
        for k, traj in enumerate(self.trajs):
            if (not opti_only or traj['type'] in ['optimal', 'integral']) and not traj['info'].startswith('ef'):
                scatter_lp = self.draw_traj(k, nolabels=nolabels)
                # Gather legends
                try:
                    if len(traj['info']) > 0:
                        self.leg_handles.append(scatter_lp)
                        self.leg_labels.append(traj['info'])
                except KeyError:
                    pass
        if self.mode_ef:
            self.draw_ef()

    def draw_ef(self):
        if not self.mode_aggregated:
            if self.ef_index is not None:
                # The time cursor corresponds to given index in front list
                i = self.ef_index(self.tcur)
                if i is not None:
                    for ef_id, fronts in self.ef_fronts.items():
                        # The front at given index may be empty : find nearest non-empty front by moving forward in time
                        while fronts[i].shape[0] == 0:
                            i += 1
                            if i >= len(fronts):
                                break
                        if i >= len(fronts):
                            break

                        points = fronts[i]
                        # Last points
                        kwargs = {'s': 5.,
                                  'color': reachability_colors['pmp']['last'],
                                  'marker': 'o',
                                  'linewidths': 1.,
                                  'zorder': ZO_TRAJS,
                                  }
                        px = np.zeros(points.shape[0])
                        px[:] = points[:, 0]
                        py = np.zeros(points.shape[0])
                        py[:] = points[:, 1]
                        if self.coords == 'gcs':
                            kwargs['latlon'] = True
                            px = RAD_TO_DEG * px
                            py = RAD_TO_DEG * py
                        if self.mode_3d:
                            pz = points[:, 2]
                            kwargs = {
                                'color': reachability_colors['pmp']['last'],
                                'zorder': ZO_TRAJS
                            }
                            scatter = self.ax.scatter(px, py, pz, **kwargs)
                        else:
                            scatter = self.ax.scatter(px, py, **kwargs)
                        self.traj_lp.append(scatter)
        else:
            for ef_id in self.ef_fronts.keys():
                if not self.ef_agg_display[ef_id]:
                    continue
                for k, traj in enumerate(self.ef_trajgroups[ef_id]):
                    self.draw_traj(k, ef_id=ef_id)

    def draw_traj(self, itr, ef_id=None, nolabels=False):
        """
        Plots the given trajectory according to selected display mode
        """
        # duration = (ts[last_index - 1] - ts[0])
        # self.t_tick = max_time / (nt_tick - 1)
        trajs = self.trajs if ef_id is None else self.ef_trajgroups[ef_id]
        points = trajs[itr]['data']
        # controls = trajs[itr]['controls']
        ts = trajs[itr]['ts']
        try:
            info = trajs[itr]['info']
        except KeyError:
            info = ''

        #TODO Change this
        label = 0
        _type = 'pmp'
        interrupted = False

        annot_label = None
        linewidth = None
        if str(info).startswith('ef_'):
            try:
                linewidth = 1.5
                annot_label = str(info).split('_')[2]
            except IndexError:
                pass
        elif len(str(info)) > 0:
            annot_label = str(info)
        if annot_label is None:
            annot_label = 'l' + str(label)

        # Lines
        if nolabels:
            p_label = None
        else:
            p_label = f'{_type} {label}'
            if p_label not in self.label_list:
                self.label_list.append(p_label)
            else:
                p_label = None

        # Determine range of indexes that can be plotted
        il = 0
        iu = ts.shape[0]

        if ef_id is None or self.mode_aggregated != 2:

            backward = False
            try:
                backward = ts[1] < ts[0]
            except IndexError:
                pass

            at_least_one = False
            for i in range(ts.shape[0]):
                if (not backward and ts[i] < self.tl) or (backward and ts[i] > self.tu):
                    il += 1
                elif (not backward and ts[i] > self.tcur) or (backward and ts[i] < self.tcur):
                    iu = i - 1
                    break
                else:
                    at_least_one = True
            # if not at_least_one:
            #     return
            if iu < il:
                return

        # Color selection
        color = {}
        if _type not in ['pmp', 'path', 'integral']:
            if len(info) > 0:
                if 'rft' in info.lower():
                    ttype = 'optimal-rft'
                else:
                    ttype = 'optimal-eft'
            else:
                ttype = _type
            color['steps'] = reachability_colors[ttype]['steps']
            color['last'] = reachability_colors[ttype]['last']
        elif info.startswith('ef'):
            if info.endswith('control'):
                ttype = 'debug'
            else:
                ttype = 'pmp'
            color['steps'] = reachability_colors[ttype]['steps']
            color['last'] = reachability_colors[ttype]['last']
        else:
            color['last'] = color['steps'] = path_colors[self.id_traj_color % len(path_colors)]
            self.id_traj_color += 1

        ls = 'solid' if 'm1' not in info else '--'  # linestyle[label % len(linestyle)]
        ls = '--' if 'optimal' in _type else ls
        ls = 'dashdot' if 'optimal' in _type and 'm1' in info else ls

        kwargs = {
            'color': color['steps'],
            'linestyle': ls,
            'label': p_label,
            'zorder': ZO_TRAJS,
            'alpha': 0.7,
            'linewidth': 2.5 if linewidth is None else linewidth,
        }
        px = points[:, 0].copy()
        py = points[:, 1].copy()
        if self.coords == 'gcs':
            kwargs['latlon'] = True
            px[:] = RAD_TO_DEG * px
            py[:] = RAD_TO_DEG * py

        if ef_id is None or self.mode_ef_display:
            self.traj_lines.append(self.ax.plot(px, py, **kwargs))

        if self.mode_energy:
            if 'energy' in trajs[itr].keys():
                c = trajs[itr]['energy'][il:iu]
                # norm = mpl_colors.Normalize(vmin=3.6e6, vmax=10*3.6e6)
                self.traj_epoints.append(
                    self.ax.scatter(px[:-1][::-1], py[:-1][::-1], c=c[::-1], cmap='tab20b', norm=self.engy_norm))

        """
        # Ticks
        if self.t_tick is not None:
            ticking_points = np.zeros((nt_tick, 2))
            ticking_controls = np.zeros((nt_tick, 2))
            nt = ts.shape[0]
            # delta_t = max_time / (nt - 1)
            k = 0
            for j, t in enumerate(ts):
                if j >= last_index:
                    break
                # if abs(t - k * t_tick) < 1.05 * (delta_t / 2.):
                if t - k * self.t_tick > -1e-3 or j == 0:
                    if k >= ticking_points.shape[0]:
                        # Reallocate ticking points on demand
                        n_p, n_d = ticking_points.shape
                        _save = np.zeros((n_p, n_d))
                        _save[:] = ticking_points
                        ticking_points = np.zeros((n_p * 10, n_d))
                        ticking_points[:n_p] = _save[:]

                        # Reallocate controls too
                        n_p, n_d = ticking_controls.shape
                        _save = np.zeros((n_p, n_d))
                        _save[:] = ticking_controls
                        ticking_controls = np.zeros((n_p * 10, n_d))
                        ticking_controls[:n_p] = _save[:]

                    ticking_points[k, :] = points[j]

                    if self.coords == 'cartesian':
                        ticking_controls[k] = np.array([cos(controls[j]), sin(controls[j])])
                    elif self.coords == 'gcs':
                        ticking_controls[k] = np.array([sin(controls[j]), cos(controls[j])])
                    k += 1
            kwargs = {'s': 5.,
                      'c': [reachability_colors[type]["time-tick"]],
                      'marker': 'o',
                      'linewidths': 1.,
                      'zorder': ZO_TRAJS}
            if self.coords == 'gcs':
                kwargs['latlon'] = True

            self.traj_ticks.append(self.ax.scatter(ticking_points[1:k, 0], ticking_points[1:k, 1], **kwargs))
        """

        # Last points
        kwargs = {'s': 10. if interrupted else 7.,
                  'color': color['last'],
                  'marker': (r'x' if interrupted else 'o'),
                  'linewidths': 1.,
                  'zorder': ZO_TRAJS,
                  'label': info,
                  }
        px = points[-1, 0]
        py = points[-1, 1]
        if self.coords == 'gcs':
            kwargs['latlon'] = True
            px = RAD_TO_DEG * px
            py = RAD_TO_DEG * py
        scatter = self.ax.scatter(px, py, **kwargs)
        self.traj_lp.append(scatter)

        # Annotation
        if self.mode_annot:
            if self.coords == 'cartesian':
                self.traj_annot.append(self.ax.annotate(str(annot_label), xy=(px, py), fontsize='x-small'))
            else:

                self.traj_annot.append(self.main_ax.annotate(str(annot_label), xy=self.map(px, py), fontsize='x-small'))

        # Heading vectors
        factor = 1. if self.coords == 'cartesian' else EARTH_RADIUS
        kwargs = {
            'color': (0.2, 0.2, 0.2, 1.0),
            'pivot': 'tail',
            'alpha': 1.0,
            'zorder': ZO_WIND_VECTORS
        }
        if self.coords == 'gcs':
            kwargs['latlon'] = True
            kwargs['width'] = 1 / 500  # factor ** 2 / 1000000
            kwargs['scale'] = 50
            # kwargs['scale'] = 1 / factor
            # kwargs['units'] = 'xy'
        elif self.coords == 'cartesian':
            kwargs['width'] = 1 / 500
            kwargs['scale'] = 50
        u = 0  # u = controls[iu]
        f = -1. if ts[1] < ts[0] else 1.
        cvec = f * np.array((np.cos(u), np.sin(u)))
        if self.coords == 'gcs':
            cvec = cvec[::-1]
        if self.mode_controls:
            self.traj_controls.append(self.ax.quiver(px, py, cvec[0], cvec[1], **kwargs))

        return scatter

    def draw_solver(self, labeling=True):
        self.clear_solver()
        kwargs = {}
        # Selecting correct plot axis
        if self.coords == 'gcs':
            kwargs['latlon'] = True
            scatterax = self.map
        else:
            scatterax = self.main_ax

        # Init point
        if self.x_init is not None:
            factor = RAD_TO_DEG if self.coords == 'gcs' else 1.
            kwargs['s'] = 100 if self.coords == 'gcs' else 70
            kwargs['color'] = 'black'
            kwargs['marker'] = 'o'
            kwargs['zorder'] = ZO_ANNOT
            self.scatter_init = scatterax.scatter(*(factor * self.x_init), **kwargs)

            # if labeling:
            #     self.mainax.annotate('Init', c, (10, 10), textcoords='offset pixels', ha='center')
            # if target_radius is not None:
            #     self.mainax.add_patch(plt.Circle(c, target_radius))
        # Target point
        if self.x_target is not None:
            factor = RAD_TO_DEG if self.coords == 'gcs' else 1.
            kwargs['s'] = 230 if self.coords == 'gcs' else 100
            kwargs['color'] = 'black'
            kwargs['marker'] = '*'
            kwargs['zorder'] = ZO_ANNOT
            self.scatter_target = scatterax.scatter(*(factor * self.x_target), **kwargs)

            # if labeling:
            #     self.mainax.annotate('Target', c, (10, 10), textcoords='offset pixels', ha='center')
            # if target_radius is not None:
            #     self.mainax.add_patch(plt.Circle(c, target_radius))

    def draw_obs(self):
        self.clear_obs()
        if self.obstacles is None:
            return
        if np.all(self.obstacles < 0.):
            return
        ma = np.ma.masked_array(self.obstacles, mask=self.obstacles < -0.5)
        if self.coords == 'cartesian':
            ax = self.main_ax
            kwargs = {}
            factor = 1.
        else:
            ax = self.ax
            kwargs = {'latlon': True}
            factor = RAD_TO_DEG
        matplotlib.rcParams['hatch.color'] = (.2, .2, .2, 1.)
        self.obs_contours.append(ax.contourf(factor * self.obs_grid[..., 0],
                                             factor * self.obs_grid[..., 1],
                                             ma,
                                             alpha=0.,
                                             # colors='grey',
                                             hatches=['//'],
                                             antialiased=True,
                                             zorder=ZO_OBS,
                                             **kwargs))

        if self.coords == 'cartesian':
            self.obs_contours.append(ax.contour(factor * self.obs_grid[..., 0],
                                                factor * self.obs_grid[..., 1],
                                                self.obstacles, [-0.5]))

    def draw_pen(self):
        self.clear_pen()
        if self.penalty is None:
            return
        if self.coords == 'cartesian':
            ax = self.main_ax
            kwargs = {}
            factor = 1.
        else:
            ax = self.ax
            kwargs = {'latlon': True}
            factor = RAD_TO_DEG
        matplotlib.rcParams['hatch.color'] = (.2, .2, .2, 1.)
        i, a = self._index('pen')
        data = (1 - a) * self.penalty['data'][i] + a * self.penalty['data'][i+1]
        self.pen_contours.append(ax.contourf(factor * self.penalty['grid'][..., 0],
                                             factor * self.penalty['grid'][..., 1],
                                             data,
                                             alpha=0.5,
                                             antialiased=True,
                                             zorder=ZO_OBS,
                                             **kwargs))

        if self.coords == 'cartesian':
            self.pen_contours.append(ax.contour(factor * self.penalty['grid'][..., 0],
                                                factor * self.penalty['grid'][..., 1],
                                                data, [-0.5]))

    def draw_all(self):
        if self.mode_3d:
            self.draw_wind()
            self.draw_trajs()
        else:
            self.draw_wind()
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
        if self.coords == 'gcs':
            self.map.drawgreatcircle(RAD_TO_DEG * self.wind['grid'][:, 0, 0].min(),
                                     RAD_TO_DEG * self.wind['grid'][0, :, 1].min(),
                                     RAD_TO_DEG * self.wind['grid'][:, -1, 0].max(),
                                     RAD_TO_DEG * self.wind['grid'][-1, :, 1].max())
            self.map.drawgreatcircle(RAD_TO_DEG * self.wind['grid'][:, 0, 0].min(),
                                     RAD_TO_DEG * self.wind['grid'][0, :, 1].max(),
                                     RAD_TO_DEG * self.wind['grid'][:, -1, 0].max(),
                                     RAD_TO_DEG * self.wind['grid'][-1, :, 1].min())

    def show_params(self, fname=None):
        fname = self.params_fname_formatted if fname is None else fname
        path = os.path.join(self.output_dir, fname)
        if not os.path.isfile(path):
            print('Parameters HTML file not found', file=sys.stderr)
        else:
            webbrowser.open(path)

    def set_output_path(self, path):
        self.output_dir = path
        basename = os.path.basename(path)
        self.output_imgpath = os.path.join(path, basename + f'.{self.img_params["format"]}')

    def set_title(self, title):
        self.title = title

    def set_coords(self, coords):
        self.coords = coords

    def reload(self):
        t_start = time.time()
        print('Reloading... ', end='')

        # Reload params
        self.import_params()
        self.load_all()

        self.draw_all()

        t_end = time.time()
        print(f'Done ({t_end - t_start:.3f}s)')

    def switch_agg(self):
        if len(self.ef_agg_display) > 0:
            for ef_id, b in self.ef_agg_display.items():
                if b:
                    break

            self.ef_agg_display[ef_id] = False
            self.ef_agg_display[(ef_id + 1) % len(self.ef_agg_display.keys())] = True

            self.draw_all()

    def toggle_agg(self):
        self.mode_aggregated = (self.mode_aggregated + 1) % 3
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

    def toggle_wind(self):
        self.mode_wind = not self.mode_wind
        self.draw_all()

    def toggle_ef(self):
        self.mode_ef = not self.mode_ef
        self.draw_all()

    def toggle_speed(self):
        self.mode_speed = not self.mode_speed
        self.draw_all()

    def toggle_annot(self):
        self.mode_annot = not self.mode_annot
        self.draw_all()

    def toggle_wind_colors(self):
        self.mode_wind_color = not self.mode_wind_color
        if self.mode_wind_color:
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

    def update_title(self):
        try:
            if self.t_tick is not None:
                ft_tick = f'{self.t_tick / 3600:.1f}h' if self.t_tick > 1800. else f'{self.t_tick:.2E}'
                self.title += f' (ticks : {ft_tick})'
            self.main_fig.suptitle(self.title)
        except TypeError:
            pass

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
        elif event.key == 'w':
            self.toggle_wind()
        elif event.key == 'h':
            self.toggle_ef()
        elif event.key == 'z':
            self.toggle_speed()
        elif event.key == 'right':
            self.increment_time()
        elif event.key == 'left':
            self.increment_time(k=-1)
        elif event.key == 'a':
            self.toggle_annot()
        elif event.key == 'l':
            self.toggle_wind_colors()
        elif event.key == 'e':
            self.toggle_energy()
        elif event.key == 'x':
            self.toggle_ef_display()

    def show(self, noparams=False, block=False):
        if not noparams:
            self.show_params()
        self.main_fig.savefig(self.output_imgpath, **self.img_params)

        self.main_fig.canvas.mpl_connect('key_press_event', self.keyboard)
        plt.show(block=block)

    def set_mode(self, flags):
        if flags is None:
            return
        if 's' in flags:
            self.switch_agg()
        if 'a' in flags:
            self.mode_aggregated = 1
        if 'A' in flags:
            self.mode_aggregated = 2
        if 'w' in flags:
            self.mode_wind = True
        if 't' in flags:
            self.toggle_rff()
        if 'h' in flags:
            self.mode_ef = False
        if 'u' in flags:
            self.rescale_wind = False
        if 'e' not in flags:
            self.mode_energy = False

    def to_movie(self, frames=50, fps=10, movie_format='apng', mini=False):
        self._info('Rendering animation')
        anim_path = os.path.join(self.output_dir, 'anim')
        if not os.path.exists(anim_path):
            os.mkdir(anim_path)
        else:
            for filename in os.listdir(anim_path):
                os.remove(os.path.join(anim_path, filename))
        for i in tqdm.tqdm(range(frames)):
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
        palette = os.path.join(self.output_dir, 'palette.png')

        if movie_format == 'gif':
            os.system(f"ffmpeg -i {first_file_in} -y -vf palettegen {palette}")
            file_out = os.path.join(self.output_dir, f'anim_s.{movie_format}')
            command = f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{pattern_in}' -i {palette} -filter_complex 'paletteuse' '{file_out}'"
            os.system(command)
            os.remove(palette)
        else:
            file_out = os.path.join(self.output_dir, f'anim.{movie_format}')
            command = f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{pattern_in}' '{file_out}'"
            # -c: v libx264 - pix_fmt yuv420p
            os.system(command)
        for filename in os.listdir(anim_path):
            if filename.endswith('.png'):
                os.remove(os.path.join(anim_path, filename))
        os.rmdir(anim_path)

    @staticmethod
    def _warn(msg):
        print(f'[Warning] {msg}', file=sys.stderr)

    @staticmethod
    def _info(msg):
        print(f'[Info] {msg}')

    @staticmethod
    def _error(msg):
        print(f'[Error] {msg}', file=sys.stderr)
        exit(1)
