#!/bin/bash

export HF_HOME="correct path to cache/"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

# Add your executable commands here
source activate correct path to envs/shared

# TASKS = "perturbed-humaneval-format-num_seeds_5,perturbed-humaneval-natgen-num_seeds_5,perturbed-humaneval-nlaugmenter-num_seeds_5,perturbed-humaneval-func_name-num_seeds_5"

# This script runs the humanevalfix benchmark of big code evaluation
python main.py \
--model correct path to models/starcoder2-3b \
--peft_model "correct path to mergerepair/out/starcoder2-3b/merged-rq1/weight-averaging/T1-T2-T3"  \
--tasks "perturbed-humaneval-natgen-num_seeds_5" \
--do_sample True \
--temperature 0.2 \
--n_samples 1 \
--batch_size 16 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt octocoder \
--save_generations_path generations_recode_octocoder.json \
--metric_output_path evaluation_recode_octocoder.json \
--max_length_generation 2048 \
--precision bf16 \
2>&1 | tee out.txt

conda deactivate