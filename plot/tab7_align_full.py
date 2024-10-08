import os
import numpy as np
import pandas as pd
from utils.parse_xls import parse_xls
from utils.get_rank import rank_columns_desc

KEY_LIST = 'GPT-4o', 'GPT-4-Turbo', 'GPT-3.5', 'Claude-3.5', 'Llama3.1-70B', 'Llama3.1-8B', \
           'Qwen2-72B', 'GLM-4', 'DeepSeek-V2', 'Moonshot', 'Yi-Large' ,'GLM-4-9B', \
           'ERNIE-4', 'Qwen2-7B'

number_mapper = lambda x, col_id: f'{x * 100:.1f}' + r'\%'
def hilight_mapper(x, col_id, rank):
    if rank == 0:
        return f'\\textbf{{{number_mapper(x, col_id)}}}'
    elif rank == 1:
        return f'\\underline{{{number_mapper(x, col_id)}}}'
    else:
        return number_mapper(x, col_id)


row_margins = 2.0, 1.5  # the first, and the rest, in ex

def get_data(key):
    res = np.zeros(9)
    cnt = np.zeros(9)
    
    def update_res(base, val):
        res[base] += val
        cnt[base] += 1
    
    try:
        df = parse_xls(key)
    except Exception as e:
        print(f'Error: {e}, when reading {key}')
        return res
    
    for index, row in df.iterrows():
        val = row['是否可用']
        
        assert isinstance(row['multi_rounds_related'], bool), f'Unknown multi_rounds_related value: {row["multi_rounds_related"]}, at index {index} of {key}'
        assert row['alignment'] in ('align', 'misalign', 'unknown'), f'Unknown alignment value: {row["alignment"]}, at index {index} of {key}'
        base = 0 if row['multi_rounds_related'] else 3

        update_res(base + (0 if row['alignment'] == 'align' else 1), val)
        update_res(base + 2, val)
        update_res(6 + (0 if row['alignment'] == 'align' else 1), val)
        update_res(8, val)
    
    return res / cnt

BEFORE_TEX = r'''\begin{table*}[htp]
\centering
\small
\begin{tabular}{|c|ccc|ccc|ccc|}
    \hline
    \multirow{2}{*}{Model} & \multicolumn{3}{c|}{Multi-turn Dependent} &  \multicolumn{3}{c|}{Multi-turn Parallel} & \multicolumn{3}{c|}{Overall} \\
        & Aligned & Misaligned & Average & Aligned & Misaligned & Average & Aligned & Misaligned & Average \\\hline
    \hline'''
AFTER_TEX = r'''\end{tabular}
\caption{Table 7}
\end{table*}'''

if __name__ == '__main__':
    data_table = np.zeros((len(KEY_LIST), 9))
    for i, key in enumerate(KEY_LIST):
        data_table[i] = get_data(key)
        
    # sort by the last column
    index = np.argsort(data_table[:, -1])[::-1]
    data_table = data_table[index]
    label_list = [KEY_LIST[i] for i in index]
        
    # find desending order for each column
    data_ranked = rank_columns_desc(data_table)

    print(BEFORE_TEX)
    print("% ====<<<<==== Auto-generated LaTeX code begin ====>>>>==== %\n")
    print(f"% --- generated by {os.path.basename(__file__)} --- %\n")

    for i, key in enumerate(label_list):
        print(r'\rule{0pt}{' + str(row_margins[1 if i else 0]) + r'ex}')
        print(f'{key} & ' 
            + ' & '.join([hilight_mapper(x, j, data_ranked[i, j]) for j, x in enumerate(data_table[i])])
            + ' \\\\')
    print('\\hline')

    print("\n% ====<<<<==== Auto-generated LaTeX code end ====>>>>==== %")
    print(AFTER_TEX)