#!/bin/bash

export HF_HOME="correct path to cache/"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

# Add your executable commands here
source activate correct path to envs/shared

# RQ 1 tasks
RQ1_TASKS=("T1-T2" "T1-T3" "T1-T4" "T1-T5" "T1-T2-T3" "T1-T2-T4" "T1-T2-T5" "T1-T3-T4" "T1-T3-T5" "T1-T4-T5" "T1-T2-T3-T4" "T1-T2-T3-T5" "T1-T2-T4-T5" "T1-T3-T4-T5" "T1-T2-T3-T4-T5")

# RQ 2 tasks
RQ2_TASKS=("T2-T3" "T2-T4" "T2-T5" "T3-T4" "T3-T5" "T4-T5" "T2-T3-T4" "T2-T3-T5" "T2-T4-T5" "T3-T4-T5" "T2-T3-T4-T5")

# RQ 3 tasks
RQ3_TASKS=('T1-T4-T2-T5' 'T1-T2-T4' 'T5-T1-T4' 'T2-T1-T5-T4' 'T5-T2-T1' 'T4-T5-T1-T2' 'T5-T1-T4-T2' 'T2-T1-T4-T5' 'T4-T1-T5' 'T1-T2-T4-T5' 
'T5-T1-T2-T4' 'T2-T5-T1' 'T4-T2-T1-T5' 'T2-T1-T4' 'T5-T4-T1' 'T4-T1-T2-T5' 'T1-T5-T4-T2' 'T5-T4-T1-T2' 'T4-T5-T1' 'T5-T1-T2' 'T4-T1-T2' 
'T1-T5-T4' 'T4-T1-T5-T2' 'T2-T4-T5-T1' 'T1-T5-T2-T4' 'T1-T4-T5-T2' 'T1-T4-T5' 'T4-T2-T5-T1' 'T2-T1-T5' 'T5-T2-T1-T4' 'T2-T4-T1-T5' 'T2-T4-T1' 
'T2-T5-T4-T1' 'T5-T2-T4-T1' 'T4-T2-T1' 'T1-T2-T5' 'T1-T5-T2' 'T1-T2-T5-T4' 'T5-T4-T2-T1' 'T4-T5-T2-T1' 'T1-T4-T2' 'T2-T5-T1-T4')

# RQ 3 tasks - Part one
RQ3_TASKS_PART_ONE=('T1-T4-T2-T5' 'T1-T2-T4' 'T5-T1-T4' 'T2-T1-T5-T4' 'T5-T2-T1' 'T4-T5-T1-T2' 'T5-T1-T4-T2' 'T2-T1-T4-T5' 'T4-T1-T5' 'T1-T2-T4-T5' 
'T5-T1-T2-T4' 'T2-T5-T1' 'T4-T2-T1-T5' 'T2-T1-T4' 'T5-T4-T1' 'T4-T1-T2-T5' 'T1-T5-T4-T2' 'T5-T4-T1-T2' 'T4-T5-T1' 'T5-T1-T2')

# RQ 3 tasks - Part two
RQ3_TASKS_PART_TWO=('T4-T1-T2' 'T1-T5-T4' 'T4-T1-T5-T2' 'T2-T4-T5-T1' 'T1-T5-T2-T4' 'T1-T4-T5-T2' 'T1-T4-T5' 'T4-T2-T5-T1' 'T2-T1-T5' 'T5-T2-T1-T4' 'T2-T4-T1-T5' 'T2-T4-T1' 
'T2-T5-T4-T1' 'T5-T2-T4-T1' 'T4-T2-T1' 'T1-T2-T5' 'T1-T5-T2' 'T1-T2-T5-T4' 'T5-T4-T2-T1' 'T4-T5-T2-T1' 'T1-T4-T2' 'T2-T5-T1-T4')

# merging method
MERGING_METHOD="dare_ties"

RQ='rq1'

MODEL='granite-3b-code-base'

# This script runs the humanevalfix benchmark of big code evaluation
for task in ${RQ1_TASKS[@]};
do
    python main.py \
    --model correct path to models/granite-3b-code-base \
    --peft_model "correct path to mergerepair/out/${MODEL}/merged-${RQ}/${MERGING_METHOD}/${task}"  \
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
    2>&1 | tee "outs_${task}.txt"
done

conda deactivate