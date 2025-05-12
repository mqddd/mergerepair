import json
import os

from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split

COMBINED_DATA_PATH = 'correct path to data/commitpackft/data.jsonl'
MESSAGE_CATEGORIES_PATH = 'correct path to data/commitpackft/message_category.json'
REFORMATTED_MESSAGE_CATEGORIES_PATH = 'correct path to data/commitpackft/reformatted_message_category.jsonl'
TASK_SPECIFIC_DATA_PATH = 'correct path to data/commitpackft/task-specific/'

TASKS = {
    'Development': ['New features', 'User interface'],
    'Program Repair': ['Bug fixes'],
    'Misc': ['Deprecation', 'Build system/tooling', 'Documentation', 'Dependencies', 'Configuration', 'Release management'],
    'Test & QA': ['Testing', 'Logging/Instrumentation'],
    'Improvement': ['Formatting/Linting', 'Refactoring/code cleanup', 'Performance improvements']
}

def init_spark():
    spark = SparkSession \
                .builder \
                .appName('task-specific-data') \
                .getOrCreate()
  
    sc = spark.sparkContext
    return spark, sc

# function to check if a string is a valid JSON
def is_valid_json(s):
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False

# reformat the json file to jsonl
def reformat_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        num_lines = sum(1 for line in f)

    with open(input_file, 'r') as infile, open(output_file, 'a') as outfile:
        buffer = ""
        for line in tqdm(infile, total=num_lines):
            if line == '[\n' or line == ']':
                continue
            
            buffer += line.strip()
            if is_valid_json(buffer[:-1]) and buffer[-1] == ',':
                outfile.write(buffer[:-1] + "\n")
                buffer = ""

def load_data(spark):
    # refomat the message_categories.json file to jsonl if not exists
    if not os.path.exists(REFORMATTED_MESSAGE_CATEGORIES_PATH):
        reformat_jsonl(MESSAGE_CATEGORIES_PATH, REFORMATTED_MESSAGE_CATEGORIES_PATH)
    
    data_df = spark.read.json(COMBINED_DATA_PATH)    
    message_categories_df = spark.read.json(REFORMATTED_MESSAGE_CATEGORIES_PATH)
    # print('Length of message_categories_df: ', message_categories_df.count())
    return data_df, message_categories_df

def create_task_specific_data(data_df, message_categories_df, spark):
    # remove duplicates from the message_categories_df using 'message' column
    message_categories_df = message_categories_df.dropDuplicates(['message'])

    # perform join on the 'message' column
    joined_df = data_df.join(message_categories_df, on='message', how='inner')

    # split the contents of the 'category' column using ','  and only keep the first element
    joined_df = joined_df.withColumn('category', split(col('category'), ',')[0])

    # create a separate dataframe for each task
    task_dfs = {}
    for task, categories in TASKS.items():
        task_dfs[task] = joined_df.filter(col('category').isin(categories))

    # print length of each task specific dataframe
    for task, df in task_dfs.items():
        print(f'Length of {task} dataframe: ', df.count())

    return task_dfs

def save_task_specific_data(task_dfs):
    for task, df in task_dfs.items():
        df.repartition(1).write.json(f'{TASK_SPECIFIC_DATA_PATH}/{task}', mode='overwrite')

if __name__ == '__main__':
    # init spark
    spark, sc = init_spark()

    # load data
    data_df, message_categories_df = load_data(spark)

    # create task specific data
    task_dfs = create_task_specific_data(data_df, message_categories_df, spark)

    # save task specific data
    save_task_specific_data(task_dfs)

    # stop spark session
    spark.stop()