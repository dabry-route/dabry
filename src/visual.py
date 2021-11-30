import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import string

from src.trajectory import Trajectory


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
        self.fig = plt.figure()
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
        self.setup_map()
        self.draw_wind()
        self.display_setup = True

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

        self.map.set_xlabel('$x$ ($m$)')
        self.map.set_ylabel('$y$ ($m$)')

        self.map.grid(visible=True, linestyle='-.', linewidth=0.5)
        self.map.tick_params(direction='in')

    def draw_wind(self):
        X, Y = np.meshgrid(np.linspace(0.1, 0.9, self.nx_wind), np.linspace(self.y_min, self.y_max, self.ny_wind))

        cartesian = np.dstack((X, Y)).reshape(-1, 2)

        res = np.array(list(map(self.windfield, list(cartesian))))
        U, V = res[:, 0], res[:, 1]

        norms = np.sqrt(U**2 + V**2)
        lognorms = np.log(np.sqrt(U**2 + V**2))

        norm = mpl.colors.Normalize()
        norm.autoscale(lognorms)
        cm = mpl.cm.jet

        sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])

        self.map.quiver(X, Y, U/norms, V/norms, lognorms, alpha=1., headwidth=2, width=0.004)
        plt.colorbar(sm, ax=[self.map], location='right')

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
        self.map.scatter(traj.points[:traj.last_index + 1, 0], traj.points[:traj.last_index + 1, 1],
                         s=s,
                         c=colors,
                         cmap=cmap,
                         label=label,
                         marker=('x' if traj.optimal else None))
        if self.mode == "full":
            for k in range(traj.points.shape[1]):
                self.state[k].scatter(traj.timestamps[:traj.last_index + 1], traj.points[:traj.last_index + 1, k], s=0.5)
            k = 0
            self.control[k].scatter(traj.timestamps[:traj.last_index + 1], traj.controls[:traj.last_index + 1], s=0.5)
