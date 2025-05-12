#!/bin/bash

export HF_HOME="correct path to cache/"

# Add your executable commands here
source activate correct path to envs/shared

python lora-tuning.py 2>&1 | tee ../out/outs.txt

conda deactivate