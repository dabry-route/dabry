import colorsys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cm
import matplotlib.ticker as mpl_ticker
from matplotlib.gridspec import GridSpec
import numpy as np
import string

from src.trajectory import Trajectory


class FontsizeConf:

    def __init__(self):
        self.fontsize = 10
        self.axes_titlesize = 10
        self.axes_labelsize = 8
        self.xtick_labelsize = 7
        self.ytick_labelsize = 7
        self.legend_fontsize = 10
        self.font_family = 'lato'
        self.mathtext_fontset = 'cm'


class Visual:
    """
    Defines all the visualization functions for the Mermoz problem
    """

    def __init__(self,
                 mode: string,
                 dim_state: int,
                 dim_control: int,
                 windfield,
                 nx_wind=16,
                 ny_wind=20):
        """
            Sets the display of the problem
            :param mode:
                "full" : for map, control over time, state over time
                "only-map" : just for the map
            :param nx_wind: number of wind anchor points in x
            :param ny_wind: number of wind anchor points in y
        """
        self.mode = mode
        self.dim_state = dim_state
        self.dim_control = dim_control
        self.windfield = windfield
        self.nx_wind = nx_wind
        self.ny_wind = ny_wind

        self.x_min, self.x_max = -.1, 1.1
        self.y_min, self.y_max = -1., 1.

        self.fig = None
        self.map = None
        self.state = []
        self.control = []
        self.display_setup = False

    def setup(self):

        fsc = FontsizeConf()
        plt.rc('font', size=fsc.fontsize)
        plt.rc('axes', titlesize=fsc.axes_titlesize)
        plt.rc('axes', labelsize=fsc.axes_labelsize)
        plt.rc('xtick', labelsize=fsc.xtick_labelsize)
        plt.rc('ytick', labelsize=fsc.ytick_labelsize)
        plt.rc('legend', fontsize=fsc.legend_fontsize)
        plt.rc('font', family=fsc.font_family)
        plt.rc('mathtext', fontset=fsc.mathtext_fontset)

        self.fig = plt.figure(num="Mermoz problem")

        if self.mode == "only-map":
            gs = GridSpec(1, 1, figure=self.fig)
            self.map = self.fig.add_subplot(gs[0, 0])
        elif self.mode == "full":
            """
            In this mode, let the map on the left hand side of the plot and plot the components of the state
            and the control on the right hand side
            """
            gs = GridSpec(self.dim_state + self.dim_control, 2, figure=self.fig)
            self.map = self.fig.add_subplot(gs[:, 0])
            for k in range(self.dim_state):
                self.state.append(self.fig.add_subplot(gs[k, 1]))
            for k in range(self.dim_control):
                self.control.append(self.fig.add_subplot(gs[k + self.dim_state, 1]))
        self.setup_cm()
        self.setup_map()
        self.draw_wind()
        self.setup_components()
        self.display_setup = True

    def setup_cm(self):
        top = np.array(colorsys.hls_to_rgb(141 / 360, .6, .81) + (1.,))
        middle = np.array(colorsys.hls_to_rgb(41 / 360, .6, .88) + (1.,))
        bottom = np.array(colorsys.hls_to_rgb(358 / 360, .6, .82) + (1.,))

        S = np.linspace(0., 1., 128)

        first = np.array([(1 - s) * top + s * middle for s in S])
        second = np.array([(1 - s) * middle + s * bottom for s in S])

        newcolors = np.vstack((first, second))
        self.cm = mpl_colors.ListedColormap(newcolors, name='Custom')

    def setup_map(self):
        """
        Sets the display of the map
        """
        self.map.axhline(y=0, color='k', linewidth=0.5)
        self.map.axvline(x=0, color='k', linewidth=0.5)
        self.map.axvline(x=1., color='k', linewidth=0.5)

        # figsize = 7.
        # plt.gcf().set_size_inches(figsize, figsize * (y_max - y_min) / (x_max - x_min))

        self.map.set_xlim(self.x_min, self.x_max)
        self.map.set_ylim(self.y_min, self.y_max)

        self.map.set_xlabel('$x\;[m]$')
        self.map.set_ylabel('$y\;[m]$')

        self.map.grid(visible=True, linestyle='-.', linewidth=0.5)
        self.map.tick_params(direction='in')

    def draw_wind(self):
        X, Y = np.meshgrid(np.linspace(-0.05, 1.05, self.nx_wind), np.linspace(self.y_min, self.y_max, self.ny_wind))

        cartesian = np.dstack((X, Y)).reshape(-1, 2)

        res = np.array(list(map(self.windfield, list(cartesian))))
        U, V = res[:, 0], res[:, 1]

        norms = np.sqrt(U ** 2 + V ** 2)
        lognorms = np.log(np.sqrt(U ** 2 + V ** 2))

        norm = mpl_colors.Normalize()
        norm.autoscale(np.array([0., 2.]))

        sm = mpl_cm.ScalarMappable(cmap=self.cm, norm=norm)

        # sm.set_array(np.array([]))

        def color_value(x):
            res = sm.to_rgba(x)
            if x > 1.:
                return res
            else:
                newres = res[0], res[1], res[2], 0.3
                return newres

        color = list(map(lambda x: color_value(x), norms))

        self.map.quiver(X, Y, U / norms, V / norms, headwidth=2, width=0.004, color=color)

        divider = make_axes_locatable(self.map)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # cb = plt.colorbar(sm, ax=[self.map], location='right')
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label('log Wind speed')
        # cb.ax.semilogy()
        # cb.ax.yaxis.set_major_formatter(mpl_ticker.LogFormatter())#mpl_ticker.FuncFormatter(lambda s, pos: (np.exp(s*np.log(10)), pos)))

    def setup_components(self):
        for state_plot in self.state:
            state_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
            state_plot.tick_params(direction='in')
        for control_plot in self.control:
            control_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
            control_plot.tick_params(direction='in')

    def plot_traj(self, traj: Trajectory, mode="default"):
        """
        Plots the given trajectory according to selected display mode
        """
        if not self.display_setup:
            self.setup()
        label = None
        s = 0.5 * np.ones(traj.last_index)
        if mode == "default":
            colors = None
            cmap = None
        elif mode == "reachability":
            colors = np.ones((traj.last_index, 4))
            colors[:, 0] *= 0.8  # R
            colors[:, 1] *= 0.8  # G
            colors[:, 2] *= 0.8  # B
            colors[:, 3] *= 0.6  # Alpha

            colors[-1, :] = np.array([0.8, 0., 0., 1.])  # Last point
            s[-1] = 2.
            cmap = plt.get_cmap("YlGn")
        else:
            raise ValueError(f"Unknown plot mode {mode}")
        self.map.scatter(traj.points[:traj.last_index, 0], traj.points[:traj.last_index, 1],
                         s=s,
                         c=colors,
                         cmap=cmap,
                         label=label,
                         marker=('x' if traj.optimal else None))
        if self.mode == "full":
            for k in range(traj.points.shape[1]):
                self.state[k].scatter(traj.timestamps[:traj.last_index + 1], traj.points[:traj.last_index + 1, k],
                                      s=0.5)
            k = 0
            self.control[k].scatter(traj.timestamps[:traj.last_index + 1], traj.controls[:traj.last_index + 1], s=0.5)
