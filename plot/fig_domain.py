import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils.parse_xls import parse_xls

to_pdf = True
color = '#B8ECBE'

TRANSLATE_TABLE = {
    "NLP" : "NLP",
    "互联网" : "Internet",
    "传媒" : "Media",
    "医疗" : "Healthcare",
    "宗教玄学" : "Religion",
    "心理情感" : "Psychology",
    "房地产" : "Real Estate",
    "招聘" : "Recruitment",
    "政务" : "Govern. Affairs",
    "教育" : "Education",
    "文化娱乐" : "Culture",
    "旅游" : "Travel",
    "汽车" : "Automobile",
    "法律" : "Law",
    "游戏" : "Gaming",
    "科技数码" : "Technology",
    "美食" : "Cuisine",
    "运动健身" : "Sports",
    "通用工作" : "General Work",
    "通用生活" : "General Life",
    "金融" : "Finance",
    "其他" : "Other",
}

def get_data(key):
    df = parse_xls(key)
    
    output = df[['领域', 'answer']].groupby('领域').count()
    output['answer'] //= 5
    return output.sort_values(by='answer', ascending=True)

data = get_data('GPT-4o')

def plot_histogram(ax, fontsize=13):
    y = data['answer']
    x = np.arange(len(y))
    
    ax.bar(x, y, color=color, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([TRANSLATE_TABLE[domain] for domain in data.index], rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel('# System Messages', fontsize=fontsize+2)
    
    ax.text(7, 35, 'Domain Distribution', fontsize=16, ha='center', va='center', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Calibri'

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300, tight_layout=True)

    plot_histogram(ax)

    file_name = 'figures/fig_domain' + ('.pdf' if to_pdf else '.png')
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    print(f'Figure saved to {file_name}.')