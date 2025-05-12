import os

ROOT = 'correct path to bigcode-evaluation-harness/out/granite-3b-code-base/merged/rq-3/weight-averaging'

RQ3_TASKS=['T1-T4-T2-T5', 'T1-T2-T4', 'T5-T1-T4', 'T2-T1-T5-T4', 'T5-T2-T1', 'T4-T5-T1-T2', 'T5-T1-T4-T2', 'T2-T1-T4-T5', 'T4-T1-T5', 'T1-T2-T4-T5', 
'T5-T1-T2-T4', 'T2-T5-T1', 'T4-T2-T1-T5', 'T2-T1-T4', 'T5-T4-T1', 'T4-T1-T2-T5', 'T1-T5-T4-T2', 'T5-T4-T1-T2', 'T4-T5-T1', 'T5-T1-T2', 'T4-T1-T2', 
'T1-T5-T4', 'T4-T1-T5-T2', 'T2-T4-T5-T1', 'T1-T5-T2-T4', 'T1-T4-T5-T2', 'T1-T4-T5', 'T4-T2-T5-T1', 'T2-T1-T5', 'T5-T2-T1-T4', 'T2-T4-T1-T5', 'T2-T4-T1', 
'T2-T5-T4-T1', 'T5-T2-T4-T1', 'T4-T2-T1', 'T1-T2-T5', 'T1-T5-T2', 'T1-T2-T5-T4', 'T5-T4-T2-T1', 'T4-T5-T2-T1', 'T1-T4-T2', 'T2-T5-T1-T4']

def get_missing_ones():
    files = os.listdir(ROOT)
    what_we_have = []
    for file in files:
        if file.startswith('logs'):
            tasks = file.split('_')[1][:-5]
            what_we_have.append(tasks)

    for task in RQ3_TASKS:
        if task not in what_we_have:
            print(task)


if __name__ == '__main__':
    get_missing_ones()