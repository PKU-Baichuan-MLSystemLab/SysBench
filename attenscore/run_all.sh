#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python main.py \
    --id 287 \
    --save_path ../output/attenscore/qwen \
    --model qwen-72b

# export CUDA_VISIBLE_DEVICES=0,1
python main.py \
    --id 287 \
    --save_path ../output/attenscore/llama31 \
    --model llama31-8b

# export CUDA_VISIBLE_DEVICES=0,1
python main.py \
    --id 287 \
    --save_path ../output/attenscore/glm \
    --model glm-9b
