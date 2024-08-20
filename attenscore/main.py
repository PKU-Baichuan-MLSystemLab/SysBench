import os
import json
import argparse
import datetime
import re

import torch
from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig

from datastore import get_data_store
from modeling_chatglm import ChatGLMForConditionalGeneration
from modeling_llama import LlamaForCausalLM
from modeling_qwen2 import Qwen2ForCausalLM

data_file_path = '../datas/system_benchmark_eval_datas.json'

# THUDM/glm-4-9b-chat, meta-llama/Meta-Llama-3.1-8B-Instruct, Qwen/Qwen2-72B-Instruct
checkpoint_paths = {
    'glm-9b': '/path/to/.cache/huggingface/hub/models--THUDM--glm-4-9b-chat/snapshots/aae8bd74af5c6dff63a49d7fbdcc89349ebf87aa/',
    'llama31-8b': '/path/to/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/',
    'qwen-72b': '/path/to/.cache/huggingface/hub/models--Qwen--Qwen2-72B-Instruct/snapshots/1af63c698f59c4235668ec9c1395468cb7cd7e79/'
}

extra_tokens = {
    'glm-9b': 2,
    'llama31-8b': 25,
    'qwen-72b': 3
}

cached_sid = set()

def load_examples(dataset_filepath):
    data = json.load(open(dataset_filepath, encoding="utf-8"))
    return data

def converation_generator(sysmeg_id):
    for entry in load_examples(data_file_path):
        if entry['system_id'] == sysmeg_id:
            print("System message ID:", sysmeg_id)
            for message in entry['messages']:
                if message['role'] == 'assistant':
                    continue # ignore ground truth
                yield message
            break
    else:
        raise ValueError(f"System message with id {sysmeg_id} not found")

def get_model_type(model_name):
    if model_name == 'glm-9b':
        return ChatGLMForConditionalGeneration
    elif model_name == 'llama31-8b':
        return LlamaForCausalLM
    elif model_name == 'qwen-72b':
        return Qwen2ForCausalLM
    else:
        raise ValueError(f"Model name {model_name} not found")

def workflow(arg, model, tokenizer, generation_config, datastore):
    if arg.id in cached_sid and not arg.ignore_cache:
        print(f"System message {arg.id} already cached, ignore")
        return
    
    datastore.clear()
    datastore.add_split_index(extra_tokens[arg.model] - 1, extra=True) # special tokens at the beginning
    
    generation_length = generation_config.max_length
    
    messages = []
    for message in converation_generator(arg.id):
        if len(messages) > 0 and messages[-1]["role"] == message["role"]:
            # concat the content
            print(messages[-1]["role"])
            assert len(messages) == 1 and messages[-1]["role"] == "user" and arg.replace
            messages[-1]["content"] += message["content"]

        messages.append(message)
        
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_length = tokenized_chat.shape[-1]
        datastore.add_split_index(input_length - 2)
        
        if message['role'] == 'system':
            print("System message:", message)
            if arg.replace:
                messages[-1]["role"] = "user"
            continue # concat next message
        
        generation_config.max_length = input_length + generation_length
        kwargs = {
            'inputs': tokenized_chat.to('cuda'),
            'generation_config' : generation_config
        }
        # if arg.seed is not None:
        #     kwargs['seed'] = arg.seed
        
        outputs = model.generate(**kwargs)
        output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        datastore.add_split_index(outputs[0].shape[-1] - 2)
        print(datetime.datetime.now(), "Output text:", output_text, flush=True)
        
        messages.append({"role": "assistant", "content": output_text})
    
    split_indices = datastore.get_split_indices()
    split_indices = [v + 1 for v in split_indices]
    
    print("Split indices:", split_indices)
    for i, idx in enumerate(split_indices):
        if i == 0:
            if datastore.has_extra:
                continue
            print(f'===== Split {i} =====', tokenizer.decode(outputs[0, :idx], skip_special_tokens=False), sep='\n')
        else:
            print(f'===== Split {i} =====', tokenizer.decode(outputs[0, split_indices[i-1]:idx], skip_special_tokens=False), sep='\n')
    
    datastore.save_data(arg.save_path, file_name=f'sid{arg.id}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=287, help="System message ID, -1 means all")
    parser.add_argument("--save_path", type=str, default='glm', help="Path to save the data")
    parser.add_argument("--model", type=str, default='glm-9b', choices=['glm-9b', 'llama31-8b', 'qwen-72b'], help="Model name")
    parser.add_argument("--ignore_cache", action='store_true', help="Ignore cache")
    parser.add_argument("--replace", action='store_true', help="Replace system message as user message")
    # parser.add_argument("--seed", type=int, default=None, help="Random seed")
    arg = parser.parse_args()
    
    
    model_cls = get_model_type(arg.model)
    checkpoint_path = checkpoint_paths[arg.model]
    
    model = model_cls.from_pretrained(checkpoint_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    generation_config = GenerationConfig.from_pretrained(checkpoint_path)
    generation_config.max_length = 8192
    print(generation_config)
    
    datastore = get_data_store()
    
    
    if arg.id == -1: # -1 means all
        from tqdm import tqdm
        # id_list = [63, 175, 187, 287, 401, 424]
        id_list = list(range(1, 501))
        
        arg.replace = False
        for id in tqdm(id_list):
            arg.id = id
            workflow(arg, model, tokenizer, generation_config, datastore)
        
        arg.replace = True
        arg.save_path += '_replace'
        for id in tqdm(id_list):
            arg.id = id
            workflow(arg, model, tokenizer, generation_config, datastore)
        
    else:
        if arg.replace:
            arg.save_path += '_replace'
        
        pattern = re.compile(r"layer_\d+_sid(\d+).npy")
        if os.path.exists(arg.save_path):
            for file in os.listdir(arg.save_path):
                match = pattern.match(file)
                if match:
                    cached_sid.add(int(match.group(1)))
        
        workflow(arg, model, tokenizer, generation_config, datastore)

if __name__ == "__main__":
    main()