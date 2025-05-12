#!/bin/bash

export HF_HOME="correct path to cache/"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

# Add your executable commands here
source activate "correct path to envs/ReCode"

# This script runs the ReCode evaluation
python run_robust.py exec func_name \
--models correct path to models/starcoder2-3b \
--datasets humaneval
2>&1 | tee out.txt

conda deactivate