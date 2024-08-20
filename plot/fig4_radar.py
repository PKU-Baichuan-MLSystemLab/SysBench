"""
======================================
Radar chart (aka spider or star chart)
======================================

This example creates a radar chart, also known as a spider or star chart [1]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.transforms import Affine2D, Bbox


from utils.parse_xls import parse_xls
from utils.generate_n_color import generate_n_colors

# KEY_LIST = 'GPT-4o', 'GPT-4-Turbo', 'GPT-3.5', 'Claude-3.5', 'ERNIE-4', \
#     'Yi-Large', 'Moonshot', 'DeepSeek-V2', 'GLM-4', 'Llama3.1-70B', \
#     'Qwen2-72B', 'Llama3.1-8B', 'Mistral-7B', 'Qwen2-7B'
KEY_LIST = 'GPT-4o', 'Claude-3.5', 'Qwen2-72B', 'GLM-4', 'Moonshot', \
           'GPT-3.5', 'ERNIE-4', 'Qwen2-7B'
LABEL_MAP = {
    # 'Total': 'total',
    'Action' : '动作约束',
    'Content': '内容约束',
    'Background': '背景约束',
    'Role': '角色约束',
    'Format': '格式约束',
    'Style': '风格约束'
}
to_pdf = True
ignore_first = False

N = len(LABEL_MAP)
M = len(KEY_LIST)

# The seed in Yanzhao's lunar calendar birthday
color_palette = generate_n_colors(M, seed=808, hue_offset_ratio=0.05, brightness_bias=0.1,
                                  saturation_mean=0.5, saturation_bias=0,
                                  shuffle='interleave' if ignore_first else 'interleave')

# https://colorkit.co/palettes/9-colors/
# color_palette = ["#538fff","#b431e6","#ff5b58","#f7ed65","#28d2ab","#fca207","#f6ccf9","#268189","#2d1a77"]

def get_data(key):
    res = np.zeros(N)
    
    try:
        df = parse_xls(key, sheet_name='不同约束类型遵循')
    except Exception as e:
        print(f'Error: {e}, when reading {key}')
        return res
    
    for i, (_, col) in enumerate(LABEL_MAP.items()):
        column = df[col]
        res[i] = column[1] / column[0]
    
    return res

data_table = np.zeros((len(KEY_LIST), N))
for i, key in enumerate(KEY_LIST):
    data_table[i] = get_data(key)

if ignore_first:
    data_table = data_table[:, 1:]
    data_firstcol = data_table[:, 0]
    del LABEL_MAP['Total']

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            deg_theta = np.degrees(theta)
            _, xticklabels = self.set_thetagrids(deg_theta, labels)
            for i, (label, angle) in enumerate(zip(xticklabels, deg_theta)):
                x,y = label.get_position()
                trans = Affine2D().scale(0.92, 0.92)        # MAGIC NUMBER
                trans = trans + Affine2D().translate(7, 6)  # MAGIC NUMBER
                lab = ax.text(x,y, label.get_text(), transform=(label.get_transform() + trans),
                            ha=label.get_ha(), va=label.get_va())
                if angle >= 90 and angle <= 270:
                    angle += 180
                lab.set_rotation(angle % 360)
            self.set_thetagrids(deg_theta, [])

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def white_space_align(labels, data, space=2):
    max_len = max(map(len, labels))
    return [label + ' ' * (max_len - len(label) + space) + f'{data[i]*100:.1f}%' for i, label in enumerate(labels)]

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Calibri'
    mpl.rcParams.update({'font.size': 12})
    
    theta = radar_factory(N - (1 if ignore_first else 0), frame='circle')

    if ignore_first:
        figsize = (5, 4.5)
    else:
        figsize = (4, 4)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300, tight_layout=True,
                           subplot_kw=dict(projection='radar'))

    aligned_labels = white_space_align(KEY_LIST, data_firstcol) if ignore_first else KEY_LIST
    indexes = list(range(M))
    if ignore_first:
        indexes = sorted(indexes, key=lambda i: data_firstcol[i], reverse=True)
    
    ax.set_rlim(0, 1)
    for color_id, i in enumerate(indexes):
        print(KEY_LIST[i], data_table[i])
        ax.plot(theta, data_table[i], color=color_palette[color_id], linewidth=1.2, linestyle='solid', label=aligned_labels[i])
        ax.fill(theta, data_table[i], facecolor=color_palette[color_id], alpha=0.05)
    ax.set_varlabels(LABEL_MAP.keys())
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], size=8)

    # add legend relative to top-left plot, top-aligned with it
    legend_kwargs = {
        'loc': (1.15, 0.17),
        'fontsize': 11,
        'labelspacing': 0.2,
        # 'frameon': False,
    }
    if ignore_first:
        legend_kwargs['loc'] = (1.15, 0.00)
        legend_kwargs['prop'] = {
            'family': 'Consolas',
            'size': 10,
        }
        # 5.2359877559829887307710723054658
        ax.text(5.105, 2.42, 'The CSR Scores', weight='bold', size=12, ha='center')
    
    legend = ax.legend(**legend_kwargs)
    
    fig.canvas.draw()

    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    left, bottom, right, top = bbox.extents
    print(left, bottom, right, top)
    
    if ignore_first:
        new_bbox = Bbox.from_extents(left + 0.2,
                                    bottom + 0.03,
                                    right + 0.25,
                                    top + 0.06)
    else:
        new_bbox = Bbox.from_extents(left + 0.08,
                                    bottom - 0.08,
                                    right + 0.28,
                                    top + 0.1)
    file_name = 'figures/fig_radar' + ('.pdf' if to_pdf else '.png')
    plt.savefig(file_name, bbox_inches=new_bbox)
    print(f'Saved to {file_name}.')
