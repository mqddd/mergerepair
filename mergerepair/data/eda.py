import pandas as pd

from transformers import AutoTokenizer

DATA_PATH = 'correct path to data/commitpackft/task-specific/Program Repair/part-00000-e3b0adff-5696-4683-a25f-0ea572039068-c000.json'
TOKENIZER_PATH = 'correct path to models/starcoder2-3b/'

# read the input json data
def read_data(path):
    data = pd.read_json(path, lines=True)
    return data

data = read_data(DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# get the tokenized length of the old_contents, new_contents, and message columns
data['old_contents_len'] = data['old_contents'].apply(lambda x: len(tokenizer(x)['input_ids']))
data['new_contents_len'] = data['new_contents'].apply(lambda x: len(tokenizer(x)['input_ids']))
data['message_len'] = data['message'].apply(lambda x: len(tokenizer(x)['input_ids']))

# get the distributaion of the tokenized length of the old_contents, new_contents, and message columns
print(data['old_contents_len'].describe())
print(data['new_contents_len'].describe())
print(data['message_len'].describe())

