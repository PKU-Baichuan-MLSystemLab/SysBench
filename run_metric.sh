#!/bin/bash

infer_model="gpt4_turbo_0409"
output="./output"
python plot/eval_output.py \
    --infer_model_name ${infer_model} \
    --output_dir ${output} \

