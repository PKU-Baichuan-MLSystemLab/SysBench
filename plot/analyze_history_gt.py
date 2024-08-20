import numpy as np
import os

from utils.parse_xls import parse_xls, TURN_NUMBER

KEY_LIST = 'GPT-4o', 'Claude-3.5', 'Llama3.1-70B', 'Llama3.1-8B', \
           'Qwen2-72B', 'ERNIE-4', 'GPT-3.5'
OUTPUT_FILE = 'output/with_gt_history_output/resultsV2.csv'

def get_data(key, root_dir='output'):
    res = np.zeros(13)
    cnt = np.zeros(13)
    
    def update_res(base, val):
        res[base] += val
        cnt[base] += 1
    
    try:
        df = parse_xls(key, root_dir=root_dir)
    except Exception as e:
        print(f'Error: {e}, when reading {key}')
        return res
    
    
    for index, row in df.iterrows():
        val = row['是否可用']
        turn = index % TURN_NUMBER
        
        assert isinstance(row['multi_rounds_related'], bool), f'Unknown multi_rounds_related value: {row["multi_rounds_related"]}, at index {index} of {key}'
        base = 0 if row['multi_rounds_related'] else 6

        update_res(base + turn, val)
        if turn > 0:
            update_res(base + 5, val)
        else:
            update_res(12, val)
        
    return np.vstack((res, cnt, res / cnt))

def parse_data(key):
    data_table = np.zeros((7, 13))
    data_table[:3, :] = get_data(key)
    data_table[3:6, :] = get_data(key, root_dir=os.path.dirname(OUTPUT_FILE))
    data_table[6] = data_table[5] - data_table[2]
    return data_table

if __name__ == '__main__':
    with open(OUTPUT_FILE, 'w') as f:
        caps = 'Corrent', 'Total', 'Ratio', 'Corrent', 'Total', 'Ratio', 'Gain'
        
        for key in KEY_LIST:
            data_table = parse_data(key)
            header = key + ','
            header += ','.join([f'R{i}' for i in range(1, 6)]) + ',AVG,'
            header += ','.join([f'R{i}' for i in range(1, 6)]) + ',AVG,AVG-R1,'
            f.write(header + '\n')
            for i in range(7):
                row = f'{caps[i]},' + ','.join([str(x) for x in data_table[i].flatten()])
                f.write(row + '\n')
            f.write('\n')
        
    print(f'Output to {OUTPUT_FILE}')