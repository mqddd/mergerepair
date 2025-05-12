#!/bin/bash

export HF_HOME="correct path to cache/"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

# Add your executable commands here
source activate correct path to envs/shared

# This script runs the humanevalfix benchmark of big code evaluation
python main.py \
--model correct path to models/starcoder2-3b \
--peft_model "correct path to mergerepair/out/starcoder2-3b/Misc/checkpoint-3405"  \
--tasks humanevalfixtests-python \
--do_sample True \
--temperature 0.2 \
--n_samples 20 \
--batch_size 16 \
--allow_code_execution \
--save_generations \
--trust_remote_code \
--prompt octocoder \
--save_generations_path generations_humanevalfixpython_octocoder.json \
--metric_output_path evaluation_humanevalfixpython_octocoder.json \
--max_length_generation 2048 \
--precision bf16 \
2>&1 | tee out.txt

conda deactivate