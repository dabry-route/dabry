import colorsys
from math import atan2
import numpy as np
from matplotlib import pyplot as plt


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


ZO_WIND_NORM = 1
ZO_WIND_VECTORS = 2
ZO_RFF = 3
ZO_WIND_ANCHORS = 4
ZO_OBS = 5
ZO_TRAJS = 6
ZO_ANNOT = 7

RAD_TO_DEG = 180. / np.pi
DEG_TO_RAD = np.pi / 180.

my_red = np.array([0.8, 0., 0., 1.])
my_red_t = np.diag((1., 1., 1., 0.2)).dot(my_red)
my_orange = np.array([1., 0.5, 0., 1.])
my_orange2 = np.array([105 / 255, 63 / 255, 0., 1.0])
my_orange_t = np.diag((1., 1., 1., 0.5)).dot(my_orange)
my_blue = np.array([0., 0., 0.8, 1.])
my_blue_t = np.diag((1., 1., 1., 0.5)).dot(my_blue)
my_dark_blue = np.array([28 / 255, 25 / 255, 117 / 255, 1.])
my_black = np.array([0., 0., 0., 1.])
my_grey1 = np.array([0.75, 0.75, 0.75, 0.6])
my_grey2 = np.array([0.7, 0.7, 0.7, 1.0])
my_grey3 = np.array([0.5, 0.5, 0.5, 1.0])
my_green = np.array([0., 0.8, 0., 1.])
my_green_t = np.diag((1., 1., 1., 0.5)).dot(my_green)
my_purple = np.array([135 / 255, 23 / 255, 176 / 255, 1.0])
my_yellow = np.array([237 / 255, 213 / 255, 31 / 255, 1.])

reachability_colors = {
    'pmp': {
        'steps': my_grey3,
        'time-tick': my_orange2,
        'last': my_black
        # 'steps': my_grey2,
        # 'time-tick': my_orange,
        # 'last': my_red
    },
    'integral': {
        'steps': my_dark_blue,
        'time-tick': my_orange2,
        'last': my_blue
    },
    'approx': {
        'steps': my_grey1,
        'time-tick': my_orange_t,
        'last': my_orange
    },
    'point': {
        'steps': my_grey1,
        'time-tick': my_orange,
        'last': my_orange
    },
    'optimal': {
        'steps': my_dark_blue,
        'time-tick': my_dark_blue,
        'last': my_dark_blue
    },
    'optimal-rft': {
        'steps': my_orange,
        'time-tick': my_orange,
        'last': my_orange
    },
    'optimal-eft': {
        'steps': my_purple,
        'time-tick': my_purple,
        'last': my_purple
    },
    'debug': {
        'steps': my_green,
        'time-tick': my_green,
        'last': my_green
    }
}

path_colors = ['r', 'b', '#fce808', 'b', 'g', 'r', 'c', 'm', 'y']

markers = ['o', '1', '2', '3', '4']

# linestyle = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 10)), (0, (5, 10)), (0, (3, 5, 1, 5))]

linestyle = ['solid']

monocolor_colors = {
    'pmp': my_red_t,
    'approx': my_orange_t,
    'point': my_blue,
    'integral': my_black
}

EARTH_RADIUS = 6378.137e3  # [m] Earth equatorial radius

# Windy default cm
CM_WINDY = [[0, [98, 113, 183, 255]],
            [1, [57, 97, 159, 255]],
            [3, [74, 148, 169, 255]],
            [5, [77, 141, 123, 255]],
            [7, [83, 165, 83, 255]],
            [9, [53, 159, 53, 255]],
            [11, [167, 157, 81, 255]],
            [13, [159, 127, 58, 255]],
            [15, [161, 108, 92, 255]],
            [17, [129, 58, 78, 255]],
            [19, [175, 80, 136, 255]],
            [21, [117, 74, 147, 255]],
            [24, [109, 97, 163, 255]],
            [27, [68, 105, 141, 255]],
            [29, [92, 144, 152, 255]],
            [36, [125, 68, 165, 255]],
            [46, [231, 215, 215, 256]],
            [51, [219, 212, 135, 256]],
            [77, [205, 202, 112, 256]],
            [104, [128, 128, 128, 255]]]

# Truncated Windy cm
# CM_WINDY_TRUNCATED = [[0, [98, 113, 183, 255]],
#                       [1, [57, 97, 159, 255]],
#                       [3, [74, 148, 169, 255]],
#                       [5, [77, 141, 123, 255]],
#                       [7, [83, 165, 83, 255]],
#                       [9, [53, 159, 53, 255]],
#                       [11, [167, 157, 81, 255]],
#                       [13, [159, 127, 58, 255]],
#                       [15, [161, 108, 92, 255]],
#                       [17, [129, 58, 78, 255]],
#                       [19, [175, 80, 136, 255]],
#                       [21, [117, 74, 147, 255]],
#                       [24, [109, 97, 163, 255]],
#                       [27, [68, 105, 141, 255]],
#                       [29, [92, 144, 152, 255]],
#                       [36, [int(1.5 * 125), int(1.5 * 68), int(1.5 * 165), 255]]]
# [36, [125, 68, 165, 255]]]

# CM_WINDY_TRUNCATED = [[1, [57, 97, 159, 255]],
#                       [3, [74, 148, 169, 255]],
#                       [5, [77, 141, 123, 255]],
#                       [7, [83, 165, 83, 255]],
#                       [9, [53, 159, 53, 255]],
#                       [11, [167, 157, 81, 255]],
#                       [13, [159, 127, 58, 255]],
#                       [15, [161, 108, 92, 255]],
#                       [17, [129, 58, 78, 255]],
#                       [19, [175, 80, 136, 255]],
#                       [21, [117, 74, 147, 255]],
#                       [24, [109, 97, 163, 255]],
#                       [27, [68, 105, 141, 255]],
#                       [30, [int(1.7 * 92), int(1.7 * 144), int(1.7 * 152), 255]]]

CM_WINDY_TRUNCATED = [[0, [98, 113, 183, 255]],
                      [1, [57, 97, 159, 255]],
                      [3, [74, 148, 169, 255]],
                      [5, [77, 141, 123, 255]],
                      [7, [83, 165, 83, 255]],
                      [9, [53, 159, 53, 255]],
                      [11, [167, 157, 81, 255]],
                      [13, [159, 127, 58, 255]],
                      [15, [161, 108, 92, 255]],
                      [17, [129, 58, 78, 255]],
                      [19, [175, 80, 136, 255]],
                      [21, [117, 74, 147, 255]],
                      [24, [109, 97, 163, 255]],
                      [27, [68, 105, 141, 255]],
                      [29, [92, 144, 152, 255]],
                      [36, [125, 68, 165, 255]],
                      [46, [231, 215, 215, 256]]]

CM_WINDY_TRUNCATED_TRUNC = [[0, [98, 113, 183, 255]],
                            [1, [57, 97, 159, 255]],
                            [3, [74, 148, 169, 255]],
                            [5, [77, 141, 123, 255]],
                            [7, [83, 165, 83, 255]],
                            [9, [53, 159, 53, 255]],
                            [11, [167, 157, 81, 255]],
                            [13, [159, 127, 58, 255]],
                            [15, [161, 108, 92, 255]],
                            [17, [129, 58, 78, 255]],
                            [19, [175, 80, 136, 255]],
                            [21, [117, 74, 147, 255]],
                            [24, [109, 97, 163, 255]]]

# Define windy cm
import matplotlib.colors as mpl_colors

cm_values = CM_WINDY_TRUNCATED_TRUNC


def lighten(c):
    hls = colorsys.rgb_to_hls(*(np.array(c[:3]) / 256.))
    hls = (hls[0], 0.5 + 0.5 * hls[1], 0.6 + 0.4 * hls[2])
    res = list(colorsys.hls_to_rgb(*hls)) + [c[3] / 256.]
    return res


def middle(x1, x2):
    # x1, x2 shall be vectors (lon, lat) in radians
    bx = np.cos(x2[1]) * np.cos(x2[0] - x1[0])
    by = np.cos(x2[1]) * np.sin(x2[0] - x1[0])
    return x1[0] + atan2(by, np.cos(x1[1]) + bx), \
           atan2(np.sin(x1[1]) + np.sin(x2[1]), np.sqrt((np.cos(x1[1]) + bx) ** 2 + by ** 2))


def desaturate(c, sat=0.):
    N, _ = c.shape
    res = np.ones((N, 4))
    for i in range(N):
        rgb = c[i][:3]
        alpha = c[i][3]
        hls = colorsys.rgb_to_hls(*rgb)
        new_rgb = colorsys.hls_to_rgb(hls[0], hls[1], sat)
        res[i, :] = new_rgb + (alpha,)
    return res


newcolors = np.array(lighten(cm_values[0][1]))
for ii in range(1, len(cm_values)):
    j_min = 10 * cm_values[ii - 1][0]
    j_max = 10 * cm_values[ii][0]
    for j in range(j_min, j_max):
        c1 = np.array(lighten(cm_values[ii - 1][1]))
        c2 = np.array(lighten(cm_values[ii][1]))
        t = (j - j_min) / (j_max - j_min)
        newcolors = np.vstack((newcolors, (1 - t) * c1 + t * c2))

windy_cm = mpl_colors.ListedColormap(newcolors, name='Windy')

colors1 = plt.cm.Blues(np.linspace(0., 1, 128))
colors21 = plt.cm.Blues_r(np.linspace(0, 1, 128))
colors22 = plt.cm.CMRmap(np.linspace(0, 0.9, 128))

# Uniform blending
# colors2 = np.column_stack((np.einsum('ij,i->ij', (colors21 + colors22)[:, :3],(1. / np.linalg.norm((colors21 + colors22)[:, :3], axis=1))), np.ones(128)))

alpha = np.linspace(0, 1, 128)
colors2 = (np.einsum('ij, i->ij', colors21, 1 - alpha) + np.einsum('ij, i->ij', colors22, alpha))

# colors2 = colors21

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
colors[:, 3] = np.ones(colors.shape[0])
custom_cm = mpl_colors.LinearSegmentedColormap.from_list('custom', colors)

custom_desat_cm = mpl_colors.LinearSegmentedColormap.from_list('custom_desat', desaturate(colors))

jet_cmap = plt.colormaps['jet']
colors = np.array(list(map(jet_cmap, np.linspace(0.1, 0.9, 128))))
jet_desat_cm = mpl_colors.LinearSegmentedColormap.from_list('jet_desat', desaturate(colors, sat=0.8))
