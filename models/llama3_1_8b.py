import requests
import random
import json
import os 


IP_with_NAMES = [
    ["<enter your server host name>", "Llama-3.1-8B-Instruct"],
]


model_name_config = {name: ip for name, ip in IP_with_NAMES}

class llama3_1_8b():
    def __init__(self, ip_with_name=None, model_name=None):
        pass 
    def __call__(self, messages, max_new_tokens=1000, temperature=0.9, ip_with_name=None, model_name=None):
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        
        if not ip_with_name and not model_name:
            ip_with_name = random.choice(IP_with_NAMES)
        elif model_name:
            ip_with_name = model_name_config[model_name], model_name
        elif ip_with_name:
            if isinstance(ip_with_name[0], str):
                ip_with_name = ip_with_name
            elif isinstance(ip_with_name[0], list): ## ip是个列表，随机选一个
                ip_with_name = random.choice(ip_with_name[0]), ip_with_name[1]
            else:
                raise ValueError('Please specify ip_with_name or model_name.')
            
        else:
            raise ValueError('Please specify ip_with_name or model_name.')
        server, model_name = ip_with_name

        parameters = {
            'max_tokens': max_new_tokens,
            'temperature': temperature,
            'top_k': 5,
            'top_p': 0.85,
            'repetition_penalty': 1.05,
            # 'use_beam_search': True
        }

        response = requests.post(
                        url = f'http://{server}/v1/chat/completions',
                        json = {
                            'model': model_name,
                            'messages': messages,
                            **parameters
                        },
                    )
        # print(11111, response.json())
        answer = response.json()['choices'][0]['message']['content']
        
        return answer


if __name__ == '__main__':
    model = llama3_1_8b()
    messages = [{"role": "system", "content": "你是百灵鸟，你是一个给人看病的医生"}, {"role": "user", "content": "你叫什么名字"}]
    print(model(messages))
