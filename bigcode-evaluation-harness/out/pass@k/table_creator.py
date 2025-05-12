import os
import pandas as pd

MODEL = 'starcoder2-3b'
ROOT_DIR = f'correct path to bigcode-evaluation-harness/out/{MODEL}/merged/'
RQS = ['rq-1', 'rq-2', 'rq-3']
METHODS = ['dare-ties', 'ties', 'weight-averaging']
# granite-3b-code-base
# base_pass1 = 16.16
# base_pass10 = 19.84

# starcoder2-3b
base_pass1 = 28.66
base_pass10 = 39.11



def gather_scores():
    for rq in RQS:
        print(rq)
        # read csv file from the root directory
        dfs = []
        for meth in METHODS:
            dfs.append(pd.read_csv(os.path.join(ROOT_DIR, rq, meth, 'scores.csv'), header=None, names=['tasks', 'pass1', 'pass10']))
        
        merged_df = dfs[0].merge(dfs[1], on='tasks', how='outer').merge(dfs[2], on='tasks', how='outer')
        # rename columns
        merged_df.columns = ['tasks', 'dare_ties_pass1', 'dare_ties_pass10', 'ties_pass1', 'ties_pass10', 'weight-averaging_pass1', 'weight-averaging_pass10']
        
        # group the tasks by the length of their name
        merged_df['task_length'] = merged_df['tasks'].apply(lambda x: len(x.split('-')))
        merged_df = merged_df.groupby('task_length', group_keys=False).apply(lambda x: x.sort_values('tasks'))

        # save the merged dataframe
        merged_df.to_csv(os.path.join(ROOT_DIR, rq, 'merged_tasks_scores.csv'), index=False)
        print("Files have been merged successfully. Output saved as 'merged_tasks.csv'.")

def create_table():
    score_columns = ['dare_ties_pass1', 'dare_ties_pass10', 'ties_pass1', 'ties_pass10', 'weight-averaging_pass1', 'weight-averaging_pass10']

    for rq in RQS:
        print(rq)
        merged_df = pd.read_csv(os.path.join(ROOT_DIR, rq, 'merged_tasks_scores.csv'))

        # iterate over the rows of the dataframe
        for index, row in merged_df.iterrows():
            # change the values of pass1 and pass10 to percentage
            for col in score_columns:
                if col.endswith('pass10'):
                    diff = round(row[col] * 100, 2) - base_pass10
                    if diff < 0:
                        merged_df.at[index, col] = str(round(row[col] * 100, 2)) + '% (' + str(round(diff, 2)) + ')'
                    else:
                        merged_df.at[index, col] = str(round(row[col] * 100, 2)) + '% (+' + str(round(diff, 2)) + ')'
                elif col.endswith('pass1'):
                    diff = round(row[col] * 100, 2) - base_pass1
                    if diff < 0:
                        merged_df.at[index, col] = str(round(row[col] * 100, 2)) + '% (' + str(round(diff, 2)) + ')'
                    else:
                        merged_df.at[index, col] = str(round(row[col] * 100, 2)) + '% (+' + str(round(diff, 2)) + ')'
        
        print(f"Creating the table for rq {rq}...")
        # print(merged_df.head())

        # save pass1 data
        pass1_df = merged_df[['tasks'] + [col for col in score_columns if col.endswith('pass1')]]
        pass1_df.to_csv(os.path.join(ROOT_DIR, rq, 'pass1_table.csv'), index=False)

        # save pass10 data
        pass10_df = merged_df[['tasks'] + [col for col in score_columns if col.endswith('pass10')]]
        pass10_df.to_csv(os.path.join(ROOT_DIR, rq, 'pass10_table.csv'), index=False)

def main():
    # gather_scores()
    create_table()

if __name__ == '__main__':
    main()