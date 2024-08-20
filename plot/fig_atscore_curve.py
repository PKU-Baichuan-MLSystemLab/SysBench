import os
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.smooth import weighted_moving_average

model_list = ['glm', 'llama31', 'qwen']
# model_list = ['qwen']

color_palette = "#F27970", "#BB9727", "#32B897"
layer_ids = {
    'glm': [19, 39],
    'llama31': [16, 31],
    'qwen': [40, 79],
}

PLOT_SID, LAYER_IDX, IS_REPLACE = 287, 0, False
BASE_DIR = './output/attenscore'
is_pdf = True

X_AXIS_SCALE = 100.
seg_indices = [1, 3, 5, 7, 9, 11]

def value_mapper(row, splits, last=None):
    if last is None:
        last = get_last_nonzero_col(row)
    return (row[1] - row[0]) / (row[last] - row[0])

def get_last_nonzero_col(row):
    for i in range(row.shape[0] - 1, -1, -1):
        if row[i] != 0:
            return i
    return 0

def load_data(file_name):
    data = np.load(file_name, allow_pickle=True).item()
    start_row = -data['split_indices'][-1]
    start_row += data['split_indices'][1]   # filter out system message
    data['data'] = data['data'][start_row:]
    return data['data'], data['split_indices']

def plot_curve(ax, model_idx, data, split_indices, seg_len, color=None, window_size=5):
    model_name = model_list[model_idx]
    
    y = [value_mapper(row, split_indices) for row in data]
    x = np.arange(len(y)).astype(np.float32)
    
    start_idx = 0
    x_start = 0.
    for i in range(seg_len.shape[1]):
        seg_length = seg_len[model_idx, i]
        target_length = seg_len[-1, i]
        
        assert seg_length.is_integer()
        end_idx = start_idx + int(seg_length)
        assert end_idx <= len(x), f'{end_idx} > {len(x)} for model {model_name}'
        
        x_seg = x[start_idx:end_idx]
        print(model_name, i, np.mean(y[start_idx:end_idx]))
        x[start_idx:end_idx] = (x_seg - x_seg[0]) * target_length / seg_length + x_start
        
        x_start += target_length
        start_idx = end_idx
    
    avg_y = np.mean(y)
    print(f'{model_name}: {avg_y:.4f}', 'split:', split_indices)
    
    y_smooth = weighted_moving_average(x, y, window_size=window_size)
    ax.axhline(avg_y, color=color, linewidth=0.5, linestyle='-.')
    ax.plot(x, y_smooth, label=model_name, color=color, linewidth=1, linestyle='-')
    return avg_y
    
def read_all_data(plot_sid=PLOT_SID, layer_idx=LAYER_IDX, is_replace=IS_REPLACE):
    data_full = {}
    split_indices_full = {}
    
    for model_name in model_list:
        fn = model_name + ('_replace' if is_replace else '')
        file_path = os.path.join(BASE_DIR, fn, 
                                 f'layer_{layer_ids[model_name][layer_idx]}_sid{plot_sid}.npy')
        data, split_indices = load_data(file_path)
        data_full[fn] = data
        split_indices_full[fn] = split_indices
    
    return data_full, split_indices_full

def do_plot(ax, window_size=5, **kwargs):
    data_full, split_indices_full = read_all_data(**kwargs)
    
    # cal average length for each segment
    data_seg_len = np.zeros((len(model_list) + 1, 5))
    for i, split_indices in enumerate(split_indices_full.values()):
        print(split_indices)
        data_seg_len[i] = np.diff(split_indices[seg_indices])
    data_seg_len[-1] = np.mean(data_seg_len[:-1], axis=0)
    data_seg_len[-1] /= np.sum(data_seg_len[-1])
    data_seg_len[-1] *= X_AXIS_SCALE
    print(data_seg_len)
    
    for i, (data, split_indices) in enumerate(zip(data_full.values(), split_indices_full.values())):
        plot_curve(ax, i, data, split_indices, data_seg_len, color=color_palette[i], window_size=window_size)
    
    x_splits = np.cumsum(data_seg_len[-1])
    for i, x_val in enumerate(x_splits):
        ax.axvline(x_val, color='lightgray', linewidth=0.5, linestyle='dotted')
        ax.text((x_val + (x_splits[i-1] if i else 0)) / 2, -0.05, f'T{i+1}', ha='center', fontsize=10)
    
    ax.set_xticks(x_splits)
    ax.set_xticklabels(['' for _ in range(5)])
    # ax.set_xticklabels([f'T{i+1}' for i in range(5)])
    
    ax.text(70, 0.4, 'Decoding Step', ha='center', fontsize=11)
    ax.arrow(50, 0.37, 40, 0, head_width=0.02, head_length=2, fc='k', ec='k')
    
    ax.text(51.5, 0.68, "System Message's Share of Total Attention Score", ha='center', 
            fontsize=9.2)

if __name__ == '__main__':
    plt.rcParams["font.family"] = "Calibri"
    mpl.rcParams.update({'font.size': 14})
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), dpi=300, tight_layout=True)
    
    do_plot(ax)
    
    hadles, labels = ax.get_legend_handles_labels()
    fig.legend(hadles, labels, loc='upper center', ncol=3, fontsize=12)
    
    file_name = f'figures/attenscore' + ('.pdf' if is_pdf else '.png')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    print(f'Figure saved to {file_name}.')
    