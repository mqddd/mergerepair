import os
import json

MODEL = 'granite-3b-code-base'
ROOT_DIR = f'correct path to bigcode-evaluation-harness/out/{MODEL}/'
MERGED_DIR = 'merged'
RQS = ['rq-1', 'rq-2', 'rq-3']
METHODS = ['dare-ties', 'ties', 'weight-averaging']
TASKS = ['Development', 'Improvement', 'Misc', 'Program Repair', 'Test & QA']
EVAL_TASK = 'humanevalfixtests-python'

def gather_merged_scores():
    for rq in RQS:
        for meth in METHODS:
            dir = os.path.join(ROOT_DIR, MERGED_DIR, rq, meth)

            # load all files in the root directory
            all_files = os.listdir(dir)

            # select only evaluation files
            all_files = [f for f in all_files if f.startswith('eval')]

            print('inserting ...!')

            # remove the previous file if exist
            if os.path.exists(os.path.join(dir, 'scores.csv')):
                os.remove(os.path.join(dir, 'scores.csv'))

            # loop over all files
            for file in all_files:
                if file.endswith('.json'):
                    with open(os.path.join(dir, file), 'r') as f:
                        data = json.load(f)
                        
                        # get the task attribute
                        task = data[EVAL_TASK]

                        # get the scores
                        pass_1 = task['pass@1']
                        pass_10 = task['pass@10']

                        # get dataset tasks
                        tasks = data['config']['peft_model'].split('/')[-1]

                        # save them in a comma seprated .csv file
                        with open(os.path.join(dir, 'scores.csv'), 'a') as f:
                            f.write(f'{tasks},{pass_1},{pass_10}\n')
    
    print('Done!')

def gather_individual_scores():
    for main_task in TASKS:
        dir = os.path.join(ROOT_DIR, main_task)

        # load all files in the root directory
        all_files = os.listdir(dir)

        # select only evaluation files
        all_files = [f for f in all_files if f.startswith('eval')]

        print('inserting ...!')

        # loop over all files
        for file in all_files:
            if file.endswith('.json'):
                with open(os.path.join(dir, file), 'r') as f:
                    data = json.load(f)
                    
                    # get the task attribute
                    task = data[EVAL_TASK]

                    # get the scores
                    pass_1 = task['pass@1']
                    pass_10 = task['pass@10']

                    # save them in a comma seprated .csv file
                    with open(os.path.join(dir, 'scores.csv'), 'w') as f:
                        f.write(f'{main_task},{pass_1},{pass_10}\n')


if __name__ == '__main__':
    gather_merged_scores()
    # gather_individual_scores()
    