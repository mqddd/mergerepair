import json
import os

from evaluate import load
from datasets import load_dataset

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# load the dataset
dataset = load_dataset('json', data_files={'test': '/scratch/st-fhendija-1/mqddd/data/humanevalpack/data/python/data/humanevalpack.jsonl'})

# read generations from json file
with open('/scratch/st-fhendija-1/mqddd/projects/bigcode-evaluation-harness/generations_recode_octocoder_perturbed-humaneval-format-num_seeds_5_checkpoint-4125.json') as f:
    generations = json.load(f)

references = dataset['test']['test']

# load the metric
code_metric = load("/scratch/st-fhendija-1/mqddd/evaluate/code_eval_octopack/code_eval_octopack/")

# evaluate the generations
results = code_metric.compute(
            references=references[:len(generations)],
            predictions=generations)

print(results)