#!/bin/bash

export HF_HOME="correct path to cache/"

# Add your executable commands here
source activate correct path to envs/shared

python merger.py 2>&1 | tee outs.txt

conda deactivate