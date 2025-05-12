
"""
ReCode: Robustness Evaluation of Code Generation Models
https://arxiv.org/abs/2212.10264
Recode is a benchmark evaluating the robustness of code generation models to code and natural language perturbations.
This task allows to run the released perturbed HumanEval benchmark, and compute the robust-pass-at-k metric.
"""
from collections import defaultdict
from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

import numpy as np
import json

_CITATION = """
@article{wang2022recode,
  title={ReCode: Robustness Evaluation of Code Generation Models},
  author={Wang, Shiqi and Li, Zheng and Qian, Haifeng and Yang, Chenghao and Wang, Zijian and Shang, Mingyue and Kumar, Varun and Tan, Samson and Ray, Baishakhi and Bhatia, Parminder and others},
  journal={arXiv preprint arXiv:2212.10264},
  year={2022}
}
"""

# https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L6
IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "using namespace std;",      
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<cmath>",
        "#include<math.h>",
        "#include<numeric>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<set>",
        "#include<map>",
        "#include<queue>",
        "#include<stack>",
        "#include<list>",
        "#include<deque>",
        "#include<boost/any.hpp>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
    ],
}

TRANSFORMATION_CATEGORIES = ["format", "func_name", "natgen", "nlaugmenter"]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {
        f"perturbed-humaneval-{category}-num_seeds_{num_seeds}": create_task(
            category, num_seeds
        )
        for category in TRANSFORMATION_CATEGORIES
        for num_seeds in range(1, 11)
    }


def create_task(category, num_seeds):
    class PerturbedHumanEval(GeneralPerturbedHumanEval):
        DATASET_NAME = category

        def __init__(self):
            super().__init__(category, num_seeds)

    return PerturbedHumanEval


class GeneralPerturbedHumanEval(Task):
    # DATASET_PATH = "RaymondLi/perturbed_humaneval"
    DATASET_PATH = "correct path to data/perturbed_humanevalfix/"

    def __init__(self, category, num_seeds):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"],
            requires_execution=True,
            category=category,
        )
        # Transformation category
        self.category = category
        self.num_seeds = num_seeds
        print(f"Loading dataset for {category} with {num_seeds} seeds")
        self.filtered_dataset = self.dataset["test"].filter(
            lambda x: x["seed"] < num_seeds
        )

    def get_dataset(self):
        """
        Returns dataset for the task or an iterable of any object, that get_prompt can handle
        Only keep the first NUM_SEEDS seeds
        """
        return self.filtered_dataset

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """

        # to be matched with the prompt used in the humanevalfix dataset from humanevalpack.py file
        prompt_base = doc["declaration"]

        instruction = f'Fix bugs in {doc["entry_point"]}.'
        context = prompt_base + doc["buggy_solution"]
        context += "\n" + doc["test"]
        inp = instruction + "\n" + context

        prompt = f'Question: {inp}\n\nAnswer:\n{prompt_base}'

        # print('Prompt:\n', prompt)

        return prompt.strip()
        # return doc["prompt"].strip()

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        Will be passed to the `process_results` function, and potentially saved.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: dict
        """
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        test_code = "\n" + test_func + "\n" + entry_point
        ref = {
            "task_id": doc["task_id"],
            "seed": doc["seed"],
            "perturbation_name": doc["perturbation_name"],
            "test_code": test_code,
        }

        return ref

    def remove_last_block(self, code):
        """
        Adapted from https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L151
        """
        for w in self.stop_words:
            if w in code:
                code = code[:code.find(w)]

        ### Find the first occassion where a chain of { } is closed
        if self.DATASET_NAME == "python":
            for i, line in enumerate(code.split("\n")):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return "\n".join(code.split("\n")[:i])
        elif self.DATASET_NAME in ["java", "js", "go", "cpp", "rust"]:
            open_brackets = 2 if self.DATASET_NAME == "java" else 1
            cut = False
            for i, c in enumerate(code):
                if c == '{':
                    open_brackets += 1
                elif c == '}':
                    open_brackets -= 1
                if open_brackets == 0:
                    code = code[:i+1]
                    cut = True
                    break
            if not cut:
                if self.DATASET_NAME == "java":
                    main_pos = code.find("public static void main")
                    if main_pos != -1:
                        code = code[:main_pos] + '}'
                    if '}' in code:
                        code = code[:code.rfind('}')] + '}'
                    if code.count('{') - 1 == code.count('}'):
                        code += "\n}"
                elif '}' in code:
                    code = code[:code.rfind('}')] + '}'
        return code


    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        # prompt = self.get_prompt(self.filtered_dataset[idx])
        # generation = generation[len(prompt) :]
        # return prompt + self._stop_at_stop_token(generation, self.stop_words)


        # adapted from HumanEvalFix task in humanevalpack.py
        doc = self.get_dataset()[idx]
        prompt = self.get_prompt(doc)        
        # if self.prompt == "diff-carper":
        #     # Only remove final stopwords like <MSG>
        #     generation = self.remove_last_block(generation[len(prompt):].rstrip())
        #     generation = prompt + generation
        #     from bigcode_eval.tasks.custom_metrics.diff_eval import split_diff
        #     # From https://github.com/CarperAI/OpenELM/blob/e6402a0696096011572152334ccbe049f89c332e/src/openelm/benchmarks/benchmark_bugs.py#L93
        #     end_of_diff = re.compile("\n[^ +-@]+")
        #     parsed: dict = split_diff(generation)
        #     if parsed and all(
        #         (s in parsed for s in ["name", "file", "message", "diff"])
        #     ):
        #         # truncate diff hunk at the first line not starting with " ", "+", "-", or "@"
        #         diff_hunk: str = end_of_diff.split(parsed["diff"])[0]
        #         # We apply diff patch loosely:
        #         #   1. it ignores the line numbers;
        #         #   2. it ignores invalid lines (not starting with " ",
        #         #   "+" or "-" and not being "@@ ... @@").
        #         # https://github.com/CarperAI/OpenELM/blob/e6402a0696096011572152334ccbe049f89c332e/src/openelm/benchmarks/benchmark_bugs.py#L162
        #         nme_idx: int = diff_hunk.find("<NME>")
        #         if nme_idx != -1:
        #             diff_hunk = diff_hunk[:nme_idx]
        #         return diff_hunk
        # else:
        #     gen = self.remove_last_block(generation[len(prompt):].rstrip())
        #     if self.prompt.startswith("diff"):
        #         return gen
        #     else:
        # Strip on the right to maintain same behavior as with get_prompt
        
        # prompt_base = self.get_prompt_base(doc)
        gen = self.remove_last_block(generation[len(prompt):].rstrip())
        prompt_base = doc["declaration"]
        return prompt_base.rstrip() + gen


    def process_results(self, generations, references, adapter_tasks=None):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(dict)
            list of dict containing refrences
        :return: dict[str: float]
        """

        # adapted from process_results of HumanEvalFix task in humanevalpack.py
        python_imports = "\n".join(IMPORT_HELPER["python"])
        generations = [
            [(python_imports + "\n" + g).strip() for g in gen] for gen in generations
        ]

        # print('generations: ', generations)

        _, detailed_results = compute_code_eval(
            references=[ref["test_code"] for ref in references],
            predictions=generations,
        )

        # Compute robust-pass-at-1. For each transformation and each prompt, we have s=5 randomly perturbed prompts.
        # With a single sample per prompt, RP@1 on a given transformation is the fraction of examples where completions
        # for all the perturbed prompts are correct.
        # With n samples per prompt, https://arxiv.org/abs/2212.10264 defines RP@1 as the average of the
        # 1/n * sum_{i=1}^n I(all s correct for generation-seed i) over all prompts.
        # An alternate could be the average of the
        # prod_{j=1}^s 1/n * sum_{i=1}^n I(j-th prompt correct for generation-seed i) over all prompts.

        # We compute RP@1 for each transformation
        # transformation -> problem -> seed -> [n results]
        transformation_problem_results = defaultdict(lambda: defaultdict(dict))
        for i, ref in enumerate(references):
            result = detailed_results[i]
            result = [x[1]["passed"] for x in result]

            assert (
                ref["seed"]
                not in transformation_problem_results[ref["perturbation_name"]][
                    ref["task_id"]
                ]
            )
            transformation_problem_results[ref["perturbation_name"]][ref["task_id"]][
                ref["seed"]
            ] = result

        # save the results in a json file
        # with open(f"results_{references[0]['perturbation_name']}_{adapter_tasks}.json", "w") as f:
        #     json.dump(detailed_results, f)

        rp1 = {}
        for transformation, problem_results in transformation_problem_results.items():
            res = {}
            res["robust-pass-at-1"] = sum(
                # results = {seed -> [n results]}
                # 1/n * sum_{i=1}^n I(all s correct for generation-seed i)
                float(all(results_)) / len(list(results.values())[0])
                for results in problem_results.values()
                for results_ in zip(*results.values())
            ) / len(problem_results)

            res["alt-robust-pass-at-1"] = sum(
                # results = {seed -> [n results]}
                # prod_{j=1}^s 1/n * sum_{i=1}^n I(j-th prompt correct for generation-seed i)
                np.prod([np.mean(results[j]) for j in results])
                for results in problem_results.values()
            ) / len(problem_results)
            rp1[transformation] = res

        # TODO: for overall-performance, a prompt is solved if correct over the s prompts for all transformation categories.
        return rp1
