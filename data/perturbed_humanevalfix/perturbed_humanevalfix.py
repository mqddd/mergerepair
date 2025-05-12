# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv
import json
import os
import re

import datasets
from datasets import Value


_CITATION = """\
@article{recode_wang2022,
  title = {ReCode: Robustness Evaluation of Code Generation Models},
  author = {Wang, Shiqi and
   Zheng, Li and
   Qian, Haifeng and
   Yang, Chenghao and
   Wang, Zijian and
   Kumar, Varun and
   Shang, Mingyue and
   Tan, Samson and
   Ray, Baishakhi and
   Bhatia, Parminder and
   Nallapati, Ramesh and
   Ramanathan, Murali Krishna and
   Roth, Dan and
   Xiang, Bing},
  doi = {10.48550/arXiv.2212.10264},
  url = {https://arxiv.org/abs/2212.10264},
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL)},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

_DESCRIPTION = """\
Perturbed version of HumanEval from: ReCode: Robustness Evaluation of Code Generation Models
"""

_HOMEPAGE = "https://github.com/amazon-science/recode"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "nlaugmenter": "nlaugmenter.tar.gz",
    "format": "format.tar.gz",
    "natgen": "natgen.tar.gz",
    "func_name": "func_name.tar.gz"
}


class PerturbedHumaneval(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="format", version=VERSION, description="Perturbations to the format of partial completions"),
        datasets.BuilderConfig(name="natgen", version=VERSION, description="NatGen perturbations on partial completions"),
        datasets.BuilderConfig(name="func_name", version=VERSION, description="Perturbations on function names"),
        datasets.BuilderConfig(name="nlaugmenter", version=VERSION, description="Perturbations on docstrings with NL-Augmenter"),
    ]

    DEFAULT_CONFIG_NAME = "func_name"

    def _info(self):
        print('here in info!!!!')
        if self.config.name in ["format", "natgen"]:  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    'task_id': Value(dtype='string'),
                    'prompt': Value(dtype='string'),
                    'entry_point': Value(dtype='string'), 'canonical_solution': Value(dtype='string'), 'test': Value(dtype='string'), 'seed': Value(dtype="int32"), 'perturbation_name': Value(dtype='string'), 'partial': Value(dtype='string'), 'declaration': Value(dtype='string'), 'buggy_solution': Value(dtype='string')
                }
            )
        elif self.config.name in ["func_name", "nlaugmenter"]:
            features = datasets.Features(
                {
                    'task_id': Value(dtype='string'), 'prompt': Value(dtype='string'), 'entry_point': Value(dtype='string'), 'canonical_solution': Value(dtype='string'), 'test': Value(dtype='string'), 'seed': Value(dtype="int32"), 'perturbation_name': Value(dtype='string'), 'declaration': Value(dtype='string'), 'buggy_solution': Value(dtype='string')
                }
            )
        else:
            raise ValueError(f"Invalid configuration name {self.config.name}")
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _URLS[self.config.name]
        print('here in split generators!!!!', self.config.name)
        # all_urls = os.listdir(urls)
        files = dl_manager.download_and_extract(urls)
        print('files:', os.listdir(files), 'what?')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "downloaded_files": dl_manager.iter_files(files),
                    # "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, downloaded_files):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        id_ = 0
        print(f'here in generate examples!!!! {downloaded_files} what?')
        # Iterate over files in .tar.gz archive
        # extract files from archive
        for file in downloaded_files:
            # find perturbation name and seed
            m = re.match(r'humanevalfix_([A-Za-z_\d]+)_s(\d+)\.jsonl', os.path.basename(file))
            assert m is not None, f"Unrecognized file-name: {file}"
            perturbation_name = m.group(1)
            seed = int(m.group(2))
            with open(file, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    example = {
                            'task_id': data['task_id'],
                            'prompt': data['prompt'],
                            'entry_point': data['entry_point'],
                            'canonical_solution': data['canonical_solution'],
                            'test': data['test'],
                            'seed': seed,
                            'perturbation_name': perturbation_name,
                            'declaration': data['declaration'],
                            'buggy_solution': data['buggy_solution']
                    }
                    if self.config.name in ["format", "natgen"]:
                        example['partial'] = data["partial"]
                    # Yields examples as (key, example) tuples
                    yield id_, example
                    id_ += 1
