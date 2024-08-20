import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import colorsys

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from analyze_history_gt import parse_data

color_palette = ["#05b9e2", "#e88290"]
KEY_LIST = 'Qwen2-72B', 'Claude-3.5', 'Llama3.1-8B', 'ERNIE-4'

to_pdf = True
xtick_labels = ['T2', 'T3', 'T4', 'T5', 'AVG']
LABEL_LSIT = ['Multi-turn Dependent', 'Multi-turn Parallel']

def add_hsv_for_color(color, h=0, s=0, v=0.1):
    old_h, old_s, old_v = colorsys.rgb_to_hsv(*mpl.colors.to_rgb(color))
    print(old_h, old_s, old_v)
    new_h = (old_h + h) % 1
    new_s = max(0, min(1, old_s + s))
    new_v = max(0, min(1, old_v + v))
    color = colorsys.hsv_to_rgb(new_h, new_s, new_v)
    return color

color_palette2 = [
    add_hsv_for_color(color_palette[0], h=0.0, s=-0.4, v=0),
    add_hsv_for_color(color_palette[1], h=0.0, s=-0.1, v=0),
]

def plot_bar(ax, key, width=0.25):
    data_table = parse_data(key) * 100

    for i in range(5):
        pad = width/2 + width / 6
        ax.bar(i - pad, data_table[6, i+1], 
               color=color_palette2[0], width=width)
        ax.bar(i + pad, data_table[6, i+7], color=color_palette2[1], width=width)
        
        ax.text(i - pad, max(0, data_table[6, i+1]) + 0.5, '%+.1f%%'%(data_table[6, i+1]),
                ha='center', rotation=90, fontsize=12, color=color_palette[0])
        ax.text(i + pad, max(0, data_table[6, i+7]) + 0.5, '%+.1f%%'%(data_table[6, i+7]), 
                ha='center', rotation=90, fontsize=12, color=color_palette[1])
        
        ax.text(i, -3, xtick_labels[i], rotation=45, ha='center', fontsize=14)
    
    tx, ty = 2, 12.5
    if key == 'Qwen2-72B':
        tx = 1.2
    ax.text(tx, ty, key, fontsize=14, ha='center', va='center', weight='bold')
    
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.xaxis.set_ticks_position('bottom')
    ax.axhline(0, color='black', linewidth=1)
    
    # for i in range(2):
    #     offset = abs(data_table[6, 6 * i])
    #     ax.axhline(offset, color=color_palette2[i], linewidth=0.5, linestyle='dotted')
        
    ax.axhline(abs(data_table[6, -1]), color='black', linewidth=0.5, linestyle='dotted')
    
    ax.tick_params(axis='x', length=0)
    
    ax.set_xticks(range(5))
    ax.set_xticklabels([])
    
    ax.set_ylim(-4, 15)

if __name__ == '__main__':
    plt.rcParams["font.family"] = "Calibri"
    mpl.rcParams.update({'font.size': 10})
    
    fig, axs = plt.subplots(2, 2, dpi=300, tight_layout=True,
                            figsize=(7.2, 4.8),
                            sharex=True, sharey=True)
    # print figure size
    # print(fig.get_size_inches())
    
    plt.subplots_adjust(wspace=0.08, hspace=0.28)
    
    
    for idx, key in enumerate(KEY_LIST):
        plot_bar(axs[idx//2, idx%2], key)
    
    patches = [mpatches.Patch(color=color_palette2[i], label=LABEL_LSIT[i]) for i in range(2)]
    patches.append(
        mlines.Line2D([], [], color='black', label="Uncertainty", linewidth=1, linestyle='dotted')
    )
    legend = fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 0.56), ncol=3, 
                        fontsize=13, borderpad=0.5, handleheight=0.7, columnspacing=1)
    fig.text(-0.01, 0.48, 'ISR Improvment(%) with Ground-truth History', 
             va='center', rotation='vertical', fontsize=15)
    
    file_name = 'figures/fig_hgt' + ('.pdf' if to_pdf else '.png')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    print(f'Figure saved to {file_name}.')
    