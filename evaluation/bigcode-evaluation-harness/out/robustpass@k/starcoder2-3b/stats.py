import os
import json

TASKS = ['Development', 'Improvement', 'Program Repair', 'Misc']
CATEGORIES = ['format', 'func_name', 'natgen', 'nlaugmenter']

ROOT = 'correct path to bigcode-evaluation-harness/out/robustpass@k/starcoder2-3b/merged/rq-1/T1-T2-T3'
METHODS = ['dare-ties', 'ties', 'weight-averaging']

def get_individual_scores(task, category):
    for i, task in enumerate(TASKS):
        if i > 2:
            continue

        for category in CATEGORIES:
            current_path = os.getcwd()
            path = os.path.join(current_path, task, category)
            files = os.listdir(path)
            scores = 0
            count = 0
            for file in files:
                if file.startswith('eval'):
                    with open(os.path.join(path, file), 'r') as f:
                        data = json.load(f)
                    
                    # get the first item in data
                    for k, key in enumerate(data):
                        if k == 0:
                            for j, item in enumerate(data[key]):
                                scores += data[key][item]['robust-pass-at-1']
                                count += 1
                    
                    result = scores / count
                    print(f'the score of task {task} in {category} is {result}')

def get_merged_scores():
    for method in METHODS:
        for cat in CATEGORIES:
            path = os.path.join(ROOT, method, cat)
            files = os.listdir(path)
            scores = 0
            count = 0
            for file in files:
                if file.startswith('eval'):
                    with open(os.path.join(path, file), 'r') as f:
                        data = json.load(f)
                    
                    # get the first item in data
                    for k, key in enumerate(data):
                        if k == 0:
                            for j, item in enumerate(data[key]):
                                scores += data[key][item]['robust-pass-at-1']
                                count += 1
                    
                    result = scores / count
                    print(f'the score of method {method} in {cat} is {result}')

def main():                     
    # get_individual_scores('Development', 'format')
    get_merged_scores()

if __name__ == "__main__":
    main()