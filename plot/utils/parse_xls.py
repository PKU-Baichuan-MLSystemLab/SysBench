import pandas as pd
import os

TOTAL_SYSTEM_ID = 500
TURN_NUMBER = 5

ENTRY_NUMBER = TOTAL_SYSTEM_ID * TURN_NUMBER

KEY_MAP = {
    'GPT-4-Turbo'   : 'gpt4_turbo_0409',
    'GPT-4o'        : 'gpt4o',
    'GPT-3.5'       : 'gpt35',
    'ERNIE-4'       : 'ernie4',
    'Moonshot'      : 'moonshot',
    'Qwen2-7B'      : 'qwen2_7b',
    'Qwen2-72B'     : 'qwen2_72b',
    'GLM-4'         : 'glm4',
    'Yi-Large'      : 'yi_large',
    'DeepSeek-V2'   : 'deepseek',
    'Claude-3.5'    : 'claude35_opus',
    'Llama3.1-70B'  : 'llama3_70b',
    'Llama3.1-8B'   : 'llama3_8b',
    'GLM-4-9B'      : 'glm_9b_client',
}

FULL_MAP = {
    'GPT-4-Turbo'   : 'GPT4-Turbo-20240409$^\dag$',
    'GPT-4o'        : 'GPT4o$^\dag$',
    'GPT-3.5'       : 'GPT3.5-Turbo-20231106$^\dag$',
    'ERNIE-4'       : 'ERNIE-4-8K-0613$^\dag$',
    'Moonshot'      : 'Moonshot-V1-8K$^\dag$',
    'Qwen2-7B'      : 'Qwen2-7B-Instruct',
    'Qwen2-72B'     : 'Qwen2-72B-Instruct',
    'GLM-4'         : 'GLM-4-0520$^\dag$',
    'DeepSeek-V2'   : 'DeepSeek-V2-0628$^\dag$',
    'Claude-3.5'    : 'Claude-3.5-Opus$^\dag$',
    'Llama3.1-70B'  : 'Llama3.1-70B-Instruct',
    'Llama3.1-8B'   : 'Llama3.1-8B-Instruct',
    'Baichuan2-13B' : 'Baichuan2-13B-Chat',
    'GLM-4-9B'      : 'GLM-4-9B-Chat',
}

def get_full_name(key):
    return FULL_MAP.get(key, key)

def parse_xls(key, sheet_name='详情', root_dir='output'):
    file_path = os.path.join(root_dir, KEY_MAP.get(key, key), f'{KEY_MAP.get(key, key)}_analysis.xlsx')
    df = pd.read_excel(file_path, sheet_name)
    if sheet_name == '详情':
        assert len(df) == ENTRY_NUMBER, f'Reading error: {len(df)} entries found, expected {ENTRY_NUMBER} entries ({file_path})'
    return df

if __name__ == '__main__':
    file_path = 'output/ernie4/ernie4_analysis.xlsx'
    data = parse_xls(file_path)
    print(data)