#!/bin/bash

export HF_HOME="correct path to cache/"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

# Add your executable commands here
source activate correct path to envs/shared

python t-sne.py

conda deactivate