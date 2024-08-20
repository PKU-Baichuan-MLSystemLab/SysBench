import os
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd

from fig_atscore_curve import do_plot as do_plot_l
from fig_atscore_replace import do_plot as do_plot_r

from fig_atscore_curve import color_palette, model_list

is_pdf = True

map_str = {
    'glm': 'GLM4-9B',
    'llama31': 'Llama3.1-8B',
    'qwen': 'Qwen2-72B',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', '-i', type=int, default=287, help='System Message ID for the plot')
    parser.add_argument('--layer', '-l', type=int, default=0, help='Middle(0) / final(1) Layer for the plot')
    parser.add_argument('--window_size', '-w', type=int, default=21, help='Window size for moving average')
    args = parser.parse_args()
    
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9), 
                            dpi=300, tight_layout=True,
                            gridspec_kw={'width_ratios': [1.4, 0.9]})
    plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1)
    
    kwargs = {
        'plot_sid': args.id,
        'layer_idx': args.layer
    }
    print(f'Plotting for System Message ID {args.id}, Layer idx {args.layer}.')
    
    do_plot_l(axs[0], args.window_size, **kwargs)
    do_plot_r(axs[1], **kwargs)
    
    axs[0].set_ylim(0.0, 0.75)
    axs[0].set_xlim(-2, 102)
    
    # fig.text(-0.01, 0.52, "System Message's Share of Total AS", va='center', rotation='vertical', fontsize=11)
    patches = [mpatches.Patch(color=color_palette[i], label=map_str[model_list[i]]) for i in range(len(model_list))]
    lines = [
        mlines.Line2D([], [], color='black', linestyle='-.', label='Average'),
        mlines.Line2D([], [], color='black', linestyle='-', label='As System'),
        mlines.Line2D([], [], color='black', linestyle='--', label='As User'),
    ]
    
    fig.text(0.55, 0.8, '(a)', ha='center', 
             fontdict={'fontsize': 12, 'font': 'Times New Roman'})
    fig.text(0.95, 0.8, '(b)', ha='center', 
             fontdict={'fontsize': 12, 'font': 'Times New Roman'})
    legend = fig.legend(handles=patches + lines, loc='upper center', ncol=6, fontsize=10,
                        bbox_to_anchor=(0.50, 1.08), columnspacing=0.5, labelspacing=0.2,
                        frameon=False,handletextpad=0.3)
    # legend.get_frame().set_alpha(None)
    # legend.get_frame().set_facecolor((0.95, 0.95, 0.95, 0.95))
    
    file_name = 'figures/atscore' + ('.pdf' if is_pdf else '.png')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.0)
    print(f'Figure saved to {file_name}.')