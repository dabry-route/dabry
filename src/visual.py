import colorsys
import string

import matplotlib
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.font_config import FontsizeConf
from src.trajectory import Trajectory, AugmentedTraj

my_red = np.array([0.8, 0., 0., 1.])
my_red_t = np.diag((1., 1., 1., 0.2)).dot(my_red)
my_orange = np.array([0.8, 0.5, 0., 1.])
my_orange_t = np.diag((1., 1., 1., 0.5)).dot(my_orange)
my_blue = np.array([0., 0., 0.8, 1.])
my_blue_t = np.diag((1., 1., 1., 0.5)).dot(my_blue)
my_black = np.array([0., 0., 0., 1.])
my_grey1 = np.array([0.75, 0.75, 0.75, 0.6])
my_grey2 = np.array([0.8, 0.8, 0.8, 0.6])
my_green = np.array([0., 0.8, 0., 1.])
my_green_t = np.diag((1., 1., 1., 0.5)).dot(my_green)

reachability_colors = {
    "pmp": {
        "steps": my_grey2,
        "time-tick": my_orange,
        "last": my_red
    },
    "integral": {
        "steps": my_grey1,
        "time-tick": my_orange,
        "last": my_blue
    },
    "approx": {
        "steps": my_grey1,
        "time-tick": my_orange_t,
        "last": my_orange
    },
    "point": {
        "steps": my_grey1,
        "time-tick": my_orange,
        "last": my_orange
    },
    "optimal": {
        "steps": my_green_t,
        "time-tick": my_green,
        "last": my_green
    },
}

monocolor_colors = {
    "pmp": my_red_t,
    "approx": my_orange_t,
    "point": my_blue,
    "integral": my_black
}

state_names = [r"$x\:[m]$", r"$y\:[m]$"]
control_names = [r"$u\:[rad]$"]

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
                 ny_wind=20,
                 title="",
                 axes_equal=True):
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

        self.x_min, self.x_max = -.1, 1.5
        self.y_min, self.y_max = -1., 1.

        self.fig = None
        self.map = None
        self.map_adjoint = None
        self.state = []
        self.control = []
        self.display_setup = False
        self.title = title
        self.axes_equal = axes_equal

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

        self.fig = plt.figure(num="Mermoz problem", constrained_layout=False)
        self.fig.subplots_adjust(
            top=0.93,
            bottom=0.08,
            left=0.075,
            right=0.93,
            hspace=0.155,
            wspace=0.13
        )
        self.fig.suptitle(self.title)

        if self.mode == "only-map":
            gs = GridSpec(1, 1, figure=self.fig)
            self.map = self.fig.add_subplot(gs[0, 0])
        elif self.mode == "full":
            """
            In this mode, let the map on the left hand side of the plot and plot the components of the state
            and the control on the right hand side
            """
            gs = GridSpec(self.dim_state + self.dim_control, 2, figure=self.fig, wspace=.25)
            self.map = self.fig.add_subplot(gs[:, 0])
            for k in range(self.dim_state):
                self.state.append(self.fig.add_subplot(gs[k, 1]))
            for k in range(self.dim_control):
                self.control.append(self.fig.add_subplot(gs[k + self.dim_state, 1]))
        elif self.mode == "full-adjoint":
            """
            In this mode, plot the state as well as the adjoint state vector
            """
            gs = GridSpec(1, 2, figure=self.fig, wspace=.25)
            self.map = self.fig.add_subplot(gs[0, 0])
            self.map_adjoint = self.fig.add_subplot(gs[0, 1])  # , projection="polar")
        self.setup_cm()
        self.setup_map()
        if self.mode == "full-adjoint":
            self.setup_map_adj()
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
        newcolors[-1] = np.array([0.4, 0., 1., 1.])
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
        if self.axes_equal:
            self.map.axis('equal')

    def setup_map_adj(self):
        """
        Sets the display of the map for the adjoint state
        """
        self.map_adjoint.set_xlim(-1.1, 1.1)
        self.map_adjoint.set_ylim(-1.1, 1.1)

        self.map_adjoint.set_xlabel('$p_x\;[s/m]$')
        self.map_adjoint.set_ylabel('$p_y\;[s/m]$')

        self.map_adjoint.grid(visible=True, linestyle='-.', linewidth=0.5)
        self.map_adjoint.tick_params(direction='in')
        if self.axes_equal:
            self.map_adjoint.axis('equal')

    def set_wind_density(self, level: int):
        """
        Sets the wind vector density in the plane.
        :param level: The density level
        """
        self.nx_wind *= level
        self.ny_wind *= level

    def draw_wind(self):
        X, Y = np.meshgrid(np.linspace(self.x_min, self.x_max, self.nx_wind),
                           np.linspace(self.y_min, self.y_max, self.ny_wind))

        cartesian = np.dstack((X, Y)).reshape(-1, 2)

        uv = np.array(list(map(self.windfield, list(cartesian))))
        U, V = uv[:, 0], uv[:, 1]

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
        cb.set_label('$|v_w|\;[m/s]$')
        # cb.ax.semilogy()
        # cb.ax.yaxis.set_major_formatter(mpl_ticker.LogFormatter())#mpl_ticker.FuncFormatter(lambda s, pos: (np.exp(s*np.log(10)), pos)))

    def setup_components(self):
        for k, state_plot in enumerate(self.state):
            state_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
            state_plot.tick_params(direction='in')
            state_plot.yaxis.set_label_position("right")
            state_plot.yaxis.tick_right()
            state_plot.set_ylabel(state_names[k])
            plt.setp(state_plot.get_xticklabels(), visible=False)

        for k, control_plot in enumerate(self.control):
            control_plot.grid(visible=True, linestyle='-.', linewidth=0.5)
            control_plot.tick_params(direction='in')
            control_plot.yaxis.set_label_position("right")
            control_plot.yaxis.tick_right()
            control_plot.set_ylabel(control_names[k])
            # Last plot
            if k == len(self.control) - 1:
                control_plot.set_xlabel(r"$t\:[s]$")

    def plot_traj(self, traj: Trajectory, color_mode="default", controls=False, **kwargs):
        """
        Plots the given trajectory according to selected display mode
        """
        if not self.display_setup:
            self.setup()
        label = None
        s = 0.5 * np.ones(traj.last_index)
        if color_mode == "default":
            colors = None
            cmap = None
        elif color_mode == "monocolor":
            colors = np.tile(monocolor_colors[traj.type], traj.last_index).reshape((traj.last_index, 4))
            cmap = None
        elif color_mode == "reachability":
            colors = np.ones((traj.last_index, 4))
            colors[:] = np.einsum("ij,j->ij", colors, reachability_colors[traj.type]["steps"])

            t_count = 0.
            for k, t in enumerate(traj.timestamps):
                if t - t_count > 0.5:
                    t_count = t
                    try:
                        colors[k] = reachability_colors[traj.type]["time-tick"]
                        s[k] = 1.5
                    except IndexError:
                        pass
            colors[-1, :] = reachability_colors[traj.type]["last"]  # Last point
            s[-1] = 2.
            cmap = plt.get_cmap("YlGn")
        elif color_mode == "reachability-enhanced":
            if "scalar_prods" not in kwargs:
                raise ValueError('Expected "scalar_prods" argument for "reachability-enhanced" plot mode')
            _cmap = plt.get_cmap('winter')
            colors = np.ones((traj.last_index, 4))
            t_count = 0.
            for k, t in enumerate(traj.timestamps):
                if t - t_count > 0.5:
                    t_count = t
                    colors[k] = reachability_colors[traj.type]["time-tick"]
                    s[k] = 2.
            for k in range(traj.last_index):
                colors[k:] = _cmap(kwargs['scalar_prods'][k])
            colors[-1, :] = reachability_colors[traj.type]["last"]  # Last point
            s *= 1.5
            s[-1] = 2.
            cmap = plt.get_cmap("YlGn")
        else:
            raise ValueError(f"Unknown plot mode {color_mode}")
        s *= 3.

        self.map.scatter(traj.points[:traj.last_index - 1, 0], traj.points[:traj.last_index - 1, 1],
                         s=s[:-1],
                         c=colors[:-1],
                         cmap=cmap,
                         label=label,
                         marker=('x' if traj.optimal else None))
        self.map.scatter(traj.points[traj.last_index - 1, 0], traj.points[traj.last_index - 1, 1],
                         s=10. if traj.interrupted else s[-1],
                         c=[colors[-1]],
                         marker=(r'x' if traj.interrupted else 'o'),
                         linewidths=1.)
        if controls:
            dt = np.mean(traj.timestamps[1:] - traj.timestamps[:-1])
            for k, point in enumerate(traj.points):
                u = traj.controls[k]
                _s = np.array([np.cos(u), np.sin(u)]) * dt
                self.map.arrow(point[0], point[1], _s[0], _s[1], width=0.0001, color=colors[k])
        if self.mode == "full":
            for k in range(traj.points.shape[1]):
                self.state[k].plot(traj.timestamps[:traj.last_index], traj.points[:traj.last_index, k])
            k = 0
            self.control[k].plot(traj.timestamps[:traj.last_index], traj.controls[:traj.last_index])
        elif self.mode == "full-adjoint":
            if isinstance(traj, AugmentedTraj):
                self.map_adjoint.scatter(traj.adjoints[:traj.last_index, 0], traj.adjoints[:traj.last_index, 1],
                                         s=s,
                                         c=colors,
                                         cmap=cmap,
                                         label=label,
                                         marker=('x' if traj.optimal else None))
