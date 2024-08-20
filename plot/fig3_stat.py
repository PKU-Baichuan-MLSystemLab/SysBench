import argparse
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox

from fig_domain import plot_histogram
from fig_constraint import plot_pie

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate figure for statistics')
    parser.add_argument('--png', action='store_true', help='Save to PDF')
    parser.add_argument('--stat_type', '-s', choices=['session', 'base'], default='session', help='Type of statistics to plot')
    args = parser.parse_args()
    
    to_pdf = not args.png
    
    plt.rcParams['font.family'] = 'Calibri'
    mpl.rcParams.update({'font.size': 10})

    # magic implementation
    fig = plt.figure(figsize=(10, 4), dpi=300, tight_layout=True)
    gs = GridSpec(2, 2, width_ratios=[6.5, 4], height_ratios=[10, 0.1],
                  wspace=0)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[:, 1])

    plot_histogram(ax1)
    plot_pie(ax2, stat_type=args.stat_type)
    
    fig.canvas.draw()

    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    left, bottom, right, top = bbox.extents
    print(left, bottom, right, top)
    new_bbox = Bbox.from_extents(left - 0.1,
                                 bottom + 0.1,
                                 right - 0.1,
                                 top + 0.1)

    file_name = 'figures/fig_stat' + ('.pdf' if to_pdf else '.png')
    plt.savefig(file_name, bbox_inches=new_bbox)
    print(f'File saved to {file_name}')