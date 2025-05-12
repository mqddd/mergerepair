import json

APR = 'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/Program Repair/logs_summerized.json'
IMPROVEMENT = 'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/Improvement/logs__summerized.json'
MISC = 'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/Misc/logs_checkpoint-3405_summerized.json'
DEVELOPMENT = 'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/Development/logs_checkpoint-5799_summerized.json'
TEST = 'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/Test & QA/logs_checkpoint-3063_summerized.json'

MERGED = 'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/merged/rq-1/weight-averaging/logs_T1-T2-T3_summerized.json'

def log_problem_details():
    path = 'correct path to data/humanevalpack/data/python/data/humanevalpack.jsonl'

    with open(path, 'r') as f:
        data = list(f)
    
    for item in data:
        item = json.loads(item)
        if item['task_id'] == 'Python/24':
            print(item)
        # print(item['problem_index'], item['problem'])

    # problem_index = 24
    # print(data[problem_index])

def log_problem_generation():
    path = 'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/merged/rq-1/weight-averaging/generations_humanevalfixpython_octocoder_humanevalfixtests-python_T1-T2-T3.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    problem_index = 24
    for item in data[problem_index]:
        print(item)
    # print(data[problem_index])

def log_solved():
    diffs = [24]
     
    # read the task file
    with open(APR, 'r') as f:
        data = json.load(f)
        
    solved = []
    for diff in diffs:
        if data[diff]['prob'] > 0.5:
            solved.append(diff)
    
    print(solved)

def check_diff():
    first_path = 'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/merged/rq-1/weight-averaging/logs_T1-T2-T3_summerized.json'
    second_path = 'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/Misc/logs_checkpoint-3405_summerized.json'

    with open(first_path, 'r') as f:
        first = json.load(f)
    
    with open(second_path, 'r') as f:
        second = json.load(f)
    
    for f, s in zip(first, second):
        if abs(f['prob'] - s['prob']) > 0.4:
            print(f['problem_index'], f['prob'], s['prob'])

def main():
    # log_problem_details()
    # log_problem_generation()
    log_solved()
    # check_diff()


if __name__ == '__main__':
    main()