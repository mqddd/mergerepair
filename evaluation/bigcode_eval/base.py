from abc import ABC, abstractmethod
from warnings import warn

from datasets import load_dataset, load_from_disk


class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True, category=None):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        # print("Loading dataset")
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        print('running evaluations for: ', category)
        try:
            # Meghdad: Changed it to load_dataset from the local file
            # self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
            
            # self.DATASET_PATH = 'correct path to data/humanevalpack/data/python/data/humanevalpack.jsonl'
            # self.dataset = load_dataset('json', data_files={'test': self.DATASET_PATH})
            # self.dataset['test'] = self.dataset['test'].select(range(5))


            print("Loading dataset from:", self.DATASET_PATH)
            # self.DATASET_PATH = 'correct path to data/perturbed_humanevalfix/perturbed_humanevalfix.py'
            # self.dataset = load_dataset(self.DATASET_PATH, 'format', trust_remote_code=True)
            self.DATASET_PATH = f'correct path to data/perturbed_humanevalfix/{category}'
            self.dataset = load_from_disk(self.DATASET_PATH)
            # self.dataset['test'] = self.dataset['test'].select(range(40))
            print("Loaded dataset in base.py")
            # self.dataset['test'] = self.dataset['train']
            # self.dataset['test'] = self.dataset['test'].select(range(5))

            # print("Loaded dataset")
            print(self.dataset)

        except Exception as e:
            warn(
                f"Loading the dataset failed with {str(e)}. This task will use a locally downloaded dataset, not from the HF hub. \
                This is expected behavior for the DS-1000 benchmark but not for other benchmarks!"
            )

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        pass

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]
