import pandas as pd
import os 

MODEL = 'starcoder2-3b'
RQ = 'rq-3'
METHOD = 'dare-ties'
BASE_PATH = f'correct path to bigcode-evaluation-harness/out/{MODEL}/merged/{RQ}/{METHOD}/'

if __name__ == '__main__':
    # read the csv file
    df = pd.read_csv(BASE_PATH + 'statistical_tests_1.csv')
    
    ##########RQ1 and RQ2##########
    # jump to the end of notationing process -> saving part

    ##########RQ3##########
    # group the tasks by the length of their name
    df['task_length'] = df['tasks'].apply(lambda x: len(x.split('-')))
    df = df.groupby('task_length', group_keys=False).apply(lambda x: x.sort_values('tasks'))

    for index, row in df.iterrows():
        LETTER = 'S'
        STARS = 1
        if row['ttest_p'] < 0.1:
            STARS = 1
        if row['ttest_p'] < 0.05:
            STARS = 2
        if row['ttest_p'] < 0.01: 
            STARS = 3
        
        if abs(row['cliff_delta']) >= 0.11:
            LETTER = 'S'
        if abs(row['cliff_delta']) >= 0.28:
            LETTER = 'M'
        if abs(row['cliff_delta']) >= 0.43:
            LETTER = 'L'
        
        notation = LETTER + ('*' * STARS)
        df.at[index, 'notation'] = notation
    
    #########RQ1 and RQ2#########
    # group by the length of the tasks and then sort each group by the tasks
    df = df.sort_values('task_length').groupby('task_length', group_keys=False).apply(lambda x: x)

    # save the dataframe
    # print(df)
    df.to_csv(os.path.join(BASE_PATH, 'notations_1.csv'), index=False)
    print(f"Statistical notations have been added successfully at {BASE_PATH}.")

    #########RQ3#########

    # group each task by its constituent tasks
    # df['lengths'] = df['tasks'].apply(lambda x: len(x.split('-')))
    # df['constituent_tasks'] = df['tasks'].apply(lambda x: '-'.join(sorted(x.split('-'))))

    # df = df.sort_values('task_length')
    # # df = df.groupby('lengths').apply(lambda x: x.sort_values('constituent_tasks'))
    # df = df.groupby('constituent_tasks', group_keys=False).apply(lambda x: x.sort_values('tasks'))
    # # df = df.sort_values('task_length')
    # # print(df[df['task_length'] == 3])
    # frames = [df[df['task_length'] == 3], df[df['task_length'] == 4]]
    # sorted_df = pd.concat(frames)
    # print(sorted_df)
    # # save
    # sorted_df.to_csv(os.path.join(BASE_PATH, 'grouped_notations_1.csv'), index=False)

            