infer_model="gpt4_turbo_0409" # change this to the model you want to evaluate
max_threads=20
python -m eval_system_bench \
    --infer_model_name ${infer_model} \
    --output_dir output \
    --max_threads ${max_threads}
