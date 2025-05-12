import pandas as pd
import os

MODELS = ['granite-3b-code-base', 'starcoder2-3b']
RQ = 'rq-3'
METHODS = ['ties', 'dare-ties', 'weight-averaging']

ROOT = 'correct path to bigcode-evaluation-harness/out/'

STARCODER_DATA = os.path.join(ROOT, MODELS[1], 'merged', 'rq-3', 'pass10_table.csv')
GRANITE_DATA = os.path.join(ROOT, MODELS[0], 'merged', 'rq-3', 'pass10_table.csv')

starcoder_pass = pd.read_csv(STARCODER_DATA)
granite_pass = pd.read_csv(GRANITE_DATA)

# pass1 threshold
# end = -6
# pass10 threshold
end = -7
starcoder_pass.columns = [(MODELS[1] + '-' + col)[:end] if col != 'tasks' else 'tasks' for col in starcoder_pass.columns]
granite_pass.columns = [(MODELS[0] + '-' + col)[:end] if col != 'tasks' else 'tasks' for col in granite_pass.columns]

# replace '_' with '-' in the columns
starcoder_pass.columns = [col.replace('_', '-') for col in starcoder_pass.columns]
granite_pass.columns = [col.replace('_', '-') for col in granite_pass.columns]
granite_pass.drop(columns=['tasks'], inplace=True)

# concatenate the data based on the tasks
df = pd.concat([starcoder_pass, granite_pass], axis=1)

print(df.columns)

# Function to extract and sort constituent tasks
def get_constituent_tasks(task_sequence):
    tasks = task_sequence.split('-')
    return '-'.join(sorted(tasks))

df['task_length'] = df['tasks'].apply(lambda x: len(x.split('-')))
df['constituent_tasks'] = df['tasks'].apply(lambda x: '-'.join(sorted(x.split('-'))))

df = df.sort_values('task_length')
df = df.groupby('constituent_tasks', group_keys=False).apply(lambda x: x.sort_values('tasks'))
frames = [df[df['task_length'] == 3], df[df['task_length'] == 4]]
sorted_df = pd.concat(frames)
sorted_df = sorted_df[['constituent_tasks', 'tasks', 'starcoder2-3b-dare-ties', 'starcoder2-3b-ties', 'starcoder2-3b-weight-averaging', 'granite-3b-code-base-dare-ties', 'granite-3b-code-base-ties', 'granite-3b-code-base-weight-averaging']]
# print(sorted_df)

for model in MODELS:
    for method in METHODS:
        BASE_PATH = f'correct path to bigcode-evaluation-harness/out/{model}/merged/{RQ}/{method}/'
        notation_df = pd.read_csv(BASE_PATH + 'grouped_notations.csv')

        # put the notation value of notation_df into the granite_code_3b_base_weight_averaging sorted_df based on the merged_tasks in sorted_df and tasks in notation_df
        for index, row in sorted_df.iterrows():
            task = row['tasks']
            notation_row = notation_df[notation_df['tasks'] == task]
            if not notation_row.empty:
                notation = notation_row['notation'].values[0]
                # print(notation)
                sorted_df.at[index, model + '-' + method] = sorted_df.at[index, model + '-' + method] + ' ' + notation

print(sorted_df)
# save the dataframe
sorted_df.to_csv(os.path.join('correct path to bigcode-evaluation-harness/out/', 'rq-3-pass@10.csv'), index=False)