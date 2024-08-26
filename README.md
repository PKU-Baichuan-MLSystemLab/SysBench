# SysBench
Code for [SysBench: Can Large Language Models Follow System Messages?](https://arxiv.org/abs/2408.10943)

## Introduction

In this section, we introduce the usage of our attached codes, including:

- SysBench's dataset.
- Running customized model on SysBench
- Reproducing figures/tables using provided data.
- Reproducing figures/tables from scratch.


## The Dataset

`datas/system_benchmark_eval_datas.json` is the dataset file, a JSON array containing 500 dialogues with system messages. For each entry, the meanings of its important JSON fields are listed in the table below.
|Field Name|Meaning|
|-|-|
|`system_id`|The ID of the system message (or dialogue)|
|`system_prompt`|The content of the system message.|
|`messages`|A JSON array containing the roles and contents of system message and the whole 5-turn conversation. The contents of the role "assistant" is the ground truth.|
|`prompt_infos`|Containing 5 JSON entries corresponding to each user instruction. For each entry, the field `alignment` denotes its alignment with the system message, and the field `criteria` is the checklists with constraint types labeled.|


## Evaluating Customized Model

This section presents the steps to evaluate a customized model on SysBench.

### Software Dependencies

Only Python (>= 3.10) and the openai (>= 1.0) package are required for the code base itself.

### Implement Interface

We provide a template model file for easily adding a new model. Suppose the customized model's name is `myModel`, copy the template by:

```sh
cd models
cp template.py myModel.py
cd ..
```

Then, rename the class name to `myModel` and implement its `__call__` method. It receives a list called `messages`, where each element is a Python dictionary containing `role` and `content` keys, representing the whole historical dialog contents for generating the next model outputs. This method should return model response contents in string format.

Some of the existing code in this directory can be used for reference. For example, `gpt4o.py` is for the OpenAI style API, `glm_9b_client.py` is for the vLLM server, while `qwen2_7b.py` is for offline inference.

### Prepare GPT-4o as Verifier

GPT-4o is used as the model-based verifier, please fill in the OpenAI Key and base URL in `gpt4o.py` to configure GPT-4o inference. Run the following command to test its usability:

```sh
python models/gpt4o.py
```

### Run Evaluation

Run the following command for evaluation:

```sh
python -m eval_system_bench \
    --infer_model_name myModel \
    --output_dir output \
    --max_threads 20
```

**Note:** It is highly recommended to use online inference and keep your `__call__` method re-entrant. Setting max_threads to 1 is required in the absence of such a guarantee.

### Calculate Metrics

After finishing the evaluation step, the detailed model and verifier outputs are both automatically stored in the `output/myModel` directory by default. To calculate the metric scores, run:

```sh
python -m eval_output \
    --infer_model_name myModel \
    --output_dir output
```

This command will output the metric scores with detailed information.


## Reproducing Results from Provided Data

Since all API keys are removed from our provided data due to privacy and anonymity requests, reproducing all results in the paper from scratch is more complicated, and we place the instructions in the next subsection. In this section, we elaborate on the steps to reproduce results with our provided raw data, which is much easier to follow.

### Software Dependencies

Python (>= 3.10), matplotlib (>= 3.9), pandas (>= 2.2), and openpyxl(>= 3.1) are required.

### Plot Figures

Run the following commands to plot figures and generate tables (in LaTeX code). We recommend installing the missing fonts for better display:

```sh
mkdir figures # create the output directory

# Plot each figure
python plot/fig3_stat.py
python plot/fig4_radar.py
python plot/fig5_hgt_histo.py
python plot/fig6_atscore.py

# Generate each table in LaTeX code
python plot/tab1_category.py
python plot/tab2_overall.py
python plot/tab3_align.py
python plot/tab4_turn.py
python plot/tab6_csr_full.py
python plot/tab7_align_full.py
```

These commands will parse the raw data in `output/` and generate figures and tables presented in the paper.

### Expected Results

All results should be **strictly consistent** with those presented in the paper.


## Reproducing Results from Scratch

To reproduce from scratch, obtaining the API keys (for all closed models) or preparing the checkpoints (for all open models) are required. Here lists the detailed steps.

### Hardware Dependencies

GPU instances are required when running open-sourced models. For the largest Qwen-72B model, we use 4× NVIDIA H100 80GB GPUs.

### Software Dependencies

transformers (>= 4.44.0) and vLLM (>= 0.5.0).

### Configure Models

Please modify **all** the model files listed in `./models` directories.
For models with public API, please fill in your public keys and the base URLs.
For open-sourced models running inference locally (i.e., Qwen family, Llama family, and GLM-4 9B), we recommend deploying a vLLM server for online serving, please check `glm_9b_client.py` for reference and modify others.

We also provide a sample script to start the vLLM server, at `servers/run_vllm_serve.sh`

### Backup Our Data (Optional)

The `output/` directory will be overwritten later.

```sh
mv output output-backup && mkdir output
```

### Exp. 1: Evaluate Models

For each model, run the following command for evaluation, please set max_threads to 1 for those without re-entrant guarantee.

```sh
python -m eval_system_bench \
    --infer_model_name <model_name> \
    --output_dir output \
    --max_threads 20
```

Then, the detailed evaluation results are available in the directory `output/<model_name>`.

### Exp. 2: Ground-truth History

We also replace the historical model response with the ground truth:

```sh
OUTDIR=output/with_gt_history_output
python -m eval_system_bench_with_gt \
    --infer_model_name <model_name> \
    --output_dir $OUTDIR \
    --max_threads 20
```

To reproduce Figure in the paper, following models should be run with the command above: `qwen2_72b`, `claude35_opus`, `ernie4` and `llama3_8b`. All results will be stored in `output/with_gt_history_output` standby.

### Exp. 3: Attention Score

To explore the distribution of attention scores, please first specify the Huggingface checkpoint paths of `glm4-9b`, `llama31-8b`, and `qwen-72b` models in Line 20-22 of `attenscore/main.py`.

Then, change the working directory to `./attenscore` and run our provided script by commands:

```sh
cd attenscore
bash run_all.sh
```

You can change the value of the `--id` flag if you want to explore another system message not presented in Figure. And set `--id` to -1 will run all 500 system messages on the current model, but very time-consuming. All results will be stored in `output/attenscore` for plotting the figure later.

### Reproduce Figures and Tables

Finally, when all experimental data are ready in the `output/`, follow the instructions for reproduction. Note that there are more available command flags for the attention score figure, run the following command for model details:

```sh
python plot/fig6_atscore -h
```

### Expected Results

Even though there exists unavoidable randomness and fluctuation, especially for closed models, all the figures and tables should statistically match the patterns shown in the paper.

## Citation

```bibtex
@article{qin2024sysbench,
  title={SysBench: Can Large Language Models Follow System Messages?},
  author={Qin, Yanzhao and Zhang, Tao and Shen, Yanjun and Luo, Wenjing and Sun, Haoze and Zhang, Yan and Qiao, Yujing and Chen, Weipeng and Zhou, Zenan and Zhang, Wentao and others},
  journal={arXiv preprint arXiv:2408.10943},
  year={2024}
}
```
