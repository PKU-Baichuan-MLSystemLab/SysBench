import re
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.parse_xls import parse_xls, TURN_NUMBER

LABEL_MAP = {
    'Action' : '动作约束',
    'Content': '内容约束',
    'Background': '背景约束',
    'Role': '角色约束',
    'Format': '格式约束',
    'Style': '风格约束',
}

sector_labels = list(LABEL_MAP.keys())
# sector_colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6']
sector_colors = ['#b7cef2', '#b9ecea', '#f2f2ca', '#f2ddb6', '#eec1c1', '#d2b6e2']

pattern = r'\d+\.\s(..约束)'

def get_data_old(key):
    res = np.zeros(len(LABEL_MAP), dtype=int)
    
    try:
        df = parse_xls(key, sheet_name='不同约束类型遵循')
    except Exception as e:
        print(f'Error: {e}, when reading {key}')
        return res
    
    for i, (_, col) in enumerate(LABEL_MAP.items()):
        res[i] = df[col][0]
    
    return res

def get_data(key):
    res = np.zeros(len(LABEL_MAP), dtype=int)
    
    try:
        df = parse_xls(key)
    except Exception as e:
        print(f'Error: {e}, when reading {key}')
        return res
    
    for index, row in df.iterrows():
        text = row['评判结果']
        turn = index % TURN_NUMBER
        
        if turn == 0:
            constraints = set()
        
        # get all constraints
        constraints.update(re.findall(pattern, text))
        
        if turn == TURN_NUMBER - 1:
            # print('session:', index//TURN_NUMBER, ', constraints:', constraints)
            for i, col in enumerate(LABEL_MAP.values()):
                if col in constraints:
                    res[i] += 1
    # print('res:', res)
    return res


to_pdf = True

def plot_pie(ax, fontsize=14, radius=0.9, stat_type='session'):
    if stat_type == 'base':
        data = get_data_old('GPT-4o')
    elif stat_type == 'session':
        data = get_data('GPT-4o')
    else:
        raise ValueError(f'Invalid stat_type: {stat_type}')
    sector_sizes = data
    
    # Properties for the wedges
    wedge_properties = {'edgecolor': 'white', 'linewidth': 1.5}  # Adjust linewidth as needed
    
    total = sum(sector_sizes)

    # Ring
    wedges, texts, autotexts = ax.pie(sector_sizes, colors=sector_colors, autopct='%1.1f%%', 
                                      startangle=90, radius=radius, wedgeprops=wedge_properties, pctdistance=0.65)

    # Rotate labels to align with wedges
    for i, (wedge, label) in enumerate(zip(wedges, autotexts)):
        if wedge.theta2 - wedge.theta1 < 40: # MAGIC NUMBER
            angle = (wedge.theta1 + wedge.theta2) / 2
            angle = angle % 360
            if angle > 90 and angle < 270:
                angle += 180
            label.set_rotation(angle)
        if wedge.theta2 - wedge.theta1 < 20: # MAGIC NUMBER
            label.set_size(fontsize - 2)
        else:
            label.set_size(fontsize)
        label.set_text(sector_labels[i] + f' ({round(sector_sizes[i]*100 / total)}%)')
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')
        label.set_color('black')

    # Draw circle in the center to make it look like a donut chart
    # centre_circle = plt.Circle((0,0),0.70,fc='white')
    # fig = plt.gcf()
    # fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    length = 2.5
    y_start, x_start = -1.5, -1.2
    ax.set_ylim(y_start, y_start + length)
    ax.set_xlim(x_start, x_start + length)
    
    ax.text(x_start + length/2, y_start + 0.25, 'Constraint Distribution', fontsize=16, ha='center', weight='bold')

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Calibri'
    mpl.rcParams.update({'font.size': 13})

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300, tight_layout=True)
    plot_pie(ax)
    
    plt.savefig('figures/fig_constraint' + ('.pdf' if to_pdf else '.png'), bbox_inches='tight', pad_inches=-0.1)