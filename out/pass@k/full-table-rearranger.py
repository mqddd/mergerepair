import pandas as pd
import os

MODELS = ['granite-3b-code-base', 'starcoder2-3b']
RQ = 'rq-2'
METHODS = ['ties', 'dare-ties', 'weight-averaging']
PASS = 'pass10_table.csv'

# concatenate the data from the two models
starcoder_pass = pd.read_csv(os.path.join('correct path to bigcode-evaluation-harness/out', 'starcoder2-3b', 'merged', RQ, PASS))
granite_pass = pd.read_csv(os.path.join('correct path to bigcode-evaluation-harness/out', 'granite-3b-code-base', 'merged', RQ, PASS))

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
data = pd.concat([starcoder_pass, granite_pass], axis=1)

# pass@10 - rq1 - both models data
for model in MODELS:
    scores_path = os.path.join('correct path to bigcode-evaluation-harness/out', model, 'merged', RQ)

    for method in METHODS:
        # read the csv file
        notation_path = None
        if RQ == 'rq-3':
            notation_path = os.path.join(scores_path, method, 'grouped_notations.csv')
        else:
            notation_path = os.path.join(scores_path, method, 'notations.csv')
        notation_df = pd.read_csv(notation_path)

        # find the correct column in data based on the model and method
        col_name = None
        for col in data.columns:
            if model + '-' + method == col:
                col_name = col
                break
        
        if col_name is None:
            # throw an exception
            raise Exception(f"Column {model + '-' + method} not found in the data.")
        # put the notation value of notation_df into the col_name of data based on the tasks in data and tasks in notation_df
        # print(data.columns)
        for index, row in data.iterrows():
            task = row['tasks']
            notation_row = notation_df[notation_df['tasks'] == task]
            if not notation_row.empty:
                notation = notation_row['notation'].values[0]
                data.at[index, col_name] = data.at[index, col_name] + ' ' + notation

# save the rearranged data
print(data.head(3))
data.to_csv(os.path.join('correct path to bigcode-evaluation-harness/out', f'{RQ}-pass10_table.csv'), index=False)