import os
import argparse

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_atscore_curve import read_all_data, \
    color_palette, model_list, value_mapper, seg_indices
    
is_pdf = True

def calc_turn(model_idx, data, split_indices, seg_len):
    model_name = model_list[model_idx]
    
    y = [value_mapper(row, split_indices) for row in data]
    x = np.arange(len(y)).astype(np.float32)
    
    start_idx = 0
    ans_y = []
    for i in range(seg_len.shape[1]):
        seg_length = seg_len[model_idx, i]
        
        assert seg_length.is_integer()
        end_idx = start_idx + int(seg_length)
        assert end_idx <= len(x), f'{end_idx} > {len(x)} for model {model_name}'
        
        ans_y.append(np.mean(y[start_idx:end_idx]))

        start_idx = end_idx

    print(model_name, ans_y)
    return ans_y

def process_data(is_replace=False, **kwargs):
    data_full, split_indices_full = read_all_data(is_replace=is_replace, **kwargs)
    
    # cal average length for each segment
    data_seg_len = np.zeros((len(model_list), 5))
    for i, split_indices in enumerate(split_indices_full.values()):
        print(split_indices)
        data_seg_len[i] = np.diff(split_indices[seg_indices])
    print(data_seg_len)
    
    res = []
    for i, (data, split_indices) in enumerate(zip(data_full.values(), split_indices_full.values())):
        res.append(calc_turn(i, data, split_indices, data_seg_len))
    
    return res

def do_plot(ax, **kwargs):
    if 'is_replace' in kwargs:
        del kwargs['is_replace']
    res_org = process_data(is_replace=False, **kwargs)
    res_rep = process_data(is_replace=True, **kwargs)
    
    x = np.arange(5) + 1
    for i, res in enumerate([res_org, res_rep]):
        for j in range(len(res)):
            ax.plot(x, res[j], label=model_list[j], 
                    color=color_palette[j], 
                    linestyle='-' if i == 0 else '--',
                    linewidth=1.2 if i == 0 else 0.5)
    
    dy = 0.035
    base_x, basey = 3.8, 0.22
    head_length_ratio = 1 / 8
    for i in range(len(model_list)):
        get_avg = lambda x: np.mean(x[i])
        avg_org = get_avg(res_org[i])
        avg_rep = get_avg(res_rep[i])
        
        if np.sign(avg_rep - avg_org) > 0:
            ax.arrow(base_x, basey + dy * i, 0, dy * 0.7, color=color_palette[i],
                     head_width=0.05, head_length=dy * head_length_ratio, 
                     length_includes_head=True, 
                     fc=color_palette[i], ec=color_palette[i])
        else:
            ax.arrow(base_x, basey + dy * (i + 0.7), 0, -dy * 0.7, color=color_palette[i],
                     head_width=0.05, head_length=dy * head_length_ratio, 
                     length_includes_head=True,
                     fc=color_palette[i], ec=color_palette[i])
        ax.text(base_x + 0.1, basey + 0.003 + dy * i, f'{(avg_rep - avg_org)*100:+.2f}%', 
                color=color_palette[i], fontdict={'fontsize': 9, 'font': 'Consolas'})
    
    rectangle = patches.Rectangle((base_x - 0.15, basey - 0.008), 1.15, dy * 3+0.04, 
                                edgecolor='black', facecolor='none', 
                                linewidth=0.8)
    ax.add_patch(rectangle)
    ax.text(base_x - 0.1, basey + dy * 3 + 0.004, 'Avg Diff.', fontsize=8, weight='bold')
    
    ax.text(3, 0.45, 'Treat System Message', ha='center', fontsize=10)
    ax.text(3, 0.415, 'as User Instruction', ha='center', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{i+1}' for i in range(5)])
    

if __name__ == '__main__':
    plt.rcParams["font.family"] = "Calibri"
    mpl.rcParams.update({'font.size': 14})
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300, tight_layout=True)
    
    do_plot(ax)
    
    hadles, labels = ax.get_legend_handles_labels()
    fig.legend(hadles[:3], labels[:3], loc='upper center', ncol=3, fontsize=12)
    
    file_name = f'figures/attenscore_r' + ('.pdf' if is_pdf else '.png')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    print(f'Figure saved to {file_name}.')
    
