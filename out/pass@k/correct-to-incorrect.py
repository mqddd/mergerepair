import os
import json

from math import comb

# directories to be used
MODEL = 'starcoder2-3b'
ROOT_DIR = f'correct path to bigcode-evaluation-harness/out/pass@k/starcoder2-3b/merged/rq-3/weight-averaging'
INTERSECT_DIR = 'intersects'
TASKS = ['Development', 'Improvement', 'Misc', 'Program Repair', 'Test & QA']
RQS = ['rq-1', 'rq-2', 'rq-3']
MERGING_METHODS = ['weight-averaging', 'ties', 'dare-ties']     
PASS_K = 1

T_TO_TASK = {
    'T1': 'Program Repair',
    'T2': 'Improvement',
    'T3': 'Misc',
    'T4': 'Development',
    'T5': 'Test & QA'
}

TASK_TO_T = {
    'Program Repair': 'T1',
    'Improvement': 'T2',
    'Misc': 'T3',
    'Development': 'T4',
    'Test & QA': 'T5'
}

def get_number_of_corrects(data):
    correct = 0
    for sample in data:
        if sample[1]['passed']:
            correct += 1
    return correct

def summerize_counts(directory):
    # read all log files in the root directory
    all_files = os.listdir(directory)
    for file in all_files:
        # if already summerized: for pass1
        if not file.endswith('summerized.json') and file.startswith('logs'):
        # if not already summerized: for pass10
        # if file.endswith('.json') and file.startswith('logs'):
            to_be_logged = []
            with open(os.path.join(directory, file), 'r') as f:
                data = json.load(f)
                probs = []
                for problem_index in data:
                    correct = 0
                    for sample in data[problem_index]:
                        if sample[1]['passed']:
                            correct += 1

                    probs.append(1-(comb(20 - correct, PASS_K)/comb(20, PASS_K)))
                    to_be_logged.append({
                        'problem_index': problem_index,
                        'correct': correct,
                        'total': 20,
                        'prob': 1-(comb(20 - correct, PASS_K)/comb(20, PASS_K))
                    })
            # save to_be_logged to a file
            with open(os.path.join(directory, f"{file.split('.')[0]}_summerized{PASS_K}.json"), 'w') as f:
                json.dump(to_be_logged, f, indent=4)  

def summerize_counts_individual(directory):
    try:
        all_dirs = os.listdir(directory)
        for dir in all_dirs:
            if dir not in TASKS:
                continue
            to_be_logged = []
            for file in os.listdir(os.path.join(directory, dir)):
                # if already summerized: for pass1
                if not file.endswith('summerized.json') and file.startswith('logs'):
                # if not already summerized: for pass10
                # if file.endswith('.json') and file.startswith('logs'):
                    with open(os.path.join(directory, dir, file), 'r') as f:
                        data = json.load(f)
                        for problem_index in data:
                            correct = 0
                            for sample in data[problem_index]:
                                if sample[1]['passed']:
                                    correct += 1
                            to_be_logged.append({
                                'problem_index': problem_index,
                                'correct': correct,
                                'total': 20,
                                'prob': 1-(comb(20 - correct, PASS_K)/comb(20, PASS_K))
                            })

                    # save to_be_logged to a file
                    with open(os.path.join(directory, dir, f"{file.split('.')[0]}_summerized_{PASS_K}.json"), 'w') as f:
                        json.dump(to_be_logged, f, indent=4)
        print('Done successfully!')
    except Exception as e:
        print(f"Error: {e}")
    
def gather_all_counts(directory):
    all_files = os.listdir(directory)
    all_counts = []
    for file in all_files:
        counter = 0
        correct_samples = []
        if file.endswith('.json') and 'summerized' in file:
            with open(os.path.join(directory, file), 'r') as f:
                data = json.load(f)
                for problem in data:
                    if problem['correct'] > 0:
                        counter += 1
                        correct_samples.append(problem['problem_index'])
            all_counts.append({'file': file, 'correct_count': counter, 'correct_samples': correct_samples})
    
    # save all_counts to a file
    with open(os.path.join(directory, f"all_counts.json"), 'w') as f:
        json.dump(all_counts, f, indent=4)

def gather_all_counts_individual(directory):
    try:
        all_dirs = os.listdir(directory)
        for dir in all_dirs:
            if dir not in TASKS:
                continue
            all_files = os.listdir(os.path.join(directory, dir))
            all_counts = []
            for file in all_files:
                counter = 0
                correct_samples = []
                if file.endswith('.json') and 'summerized' in file:
                    with open(os.path.join(directory, dir, file), 'r') as f:
                        data = json.load(f)
                        for problem in data:
                            if problem['correct'] > 0:
                                counter += 1
                                correct_samples.append(problem['problem_index'])
                    all_counts.append({'file': file, 'correct_count': counter, 'correct_samples': correct_samples})
            
            # save all_counts to a file
            with open(os.path.join(directory, dir, f"all_counts.json"), 'w') as f:
                json.dump(all_counts, f, indent=4)
        print('Done successfully!')
    except Exception as e:
        print(f"Error: {e}")

def generate_summaries_and_counts():
    for rq in RQS:
        if rq == 'rq-3':
            continue
        for method in MERGING_METHODS:
            directory = os.path.join(ROOT_DIR, rq, method)
            summerize_counts(directory)
            gather_all_counts(directory)

# we do not require this as we need to analyze the intersections between the merged tasks and individual ones
def get_correct_intersections_between_merged_tasks():
    for rq in RQS:
        if rq == 'rq-3':
            continue
        for method in MERGING_METHODS:
            directory = os.path.join(ROOT_DIR, rq, method)
            print(f"RQ: {rq}, Method: {method}")
            intersection_logs = []
            with open(os.path.join(directory, 'all_counts.json'), 'r') as f:
                data = json.load(f)
                for i in range(len(data)):
                    for j in range(i+1, len(data)):
                        intersection = set(data[i]['correct_samples']).intersection(set(data[j]['correct_samples']))
                        difference = set(data[i]['correct_samples']).symmetric_difference(set(data[j]['correct_samples']))
                        print('-----------------------------------')
                        print(f"len of correct samples in {data[i]['file'].split('_')[1]}: {len(data[i]['correct_samples'])}")
                        print(f"len of correct samples in {data[j]['file'].split('_')[1]}: {len(data[j]['correct_samples'])}")
                        print(f"Intersection between {data[i]['file'].split('_')[1]} and {data[j]['file'].split('_')[1]}: {len(intersection)}")
                        print(f"Difference between {data[i]['file'].split('_')[1]} and {data[j]['file'].split('_')[1]}: {len(difference)}, which are: {difference}")
                        print('-----------------------------------')
                        intersection_logs.append({
                            'merged_tasks_1': data[i]['file'].split('_')[1],
                            'merged_tasks_2': data[j]['file'].split('_')[1],
                            'intersections': len(intersection),
                            'differences': len(difference),
                            'difference_set': list(difference)
                        })

            with open(os.path.join(directory, 'intersection_logs.json'), 'w') as f:
                json.dump(intersection_logs, f, indent=4)

def get_intersections(data1, data2, task1, task2):
    data1_correct_samples = set(data1['correct_samples'])
    data2_correct_samples = set(data2['correct_samples'])
    intersection = data1_correct_samples.intersection(data2_correct_samples)
    differece = data1_correct_samples.symmetric_difference(data2_correct_samples)
    data1_diff = data1_correct_samples.difference(intersection)
    data2_diff = data2_correct_samples.difference(intersection)
    return {
        'task_1': task1,
        'task_2': task2,
        'intersections': len(intersection),
        'intersection_set': list(intersection),
        'task_1_diff': list(data1_diff),
        'len_task_1_diff': len(data1_diff),
        'task_2_diff': list(data2_diff),
        'len_task_2_diff': len(data2_diff),
        'differences': len(differece),
    }

def log_all_intersections(directory, rq, merging_method):
    # get individual tasks data
    all_dirs = os.listdir(directory)
    individual_tasks_data = {}
    for dir in all_dirs:
        if dir not in TASKS:
            continue
        with open(os.path.join(directory, dir, 'all_counts.json'), 'r') as f:
            data = json.load(f)
            # there should be only one item in the list of individual tasks data
            t_task = TASK_TO_T[dir]
            individual_tasks_data[t_task] = data[0]
    
    # get merged tasks data
    merged_tasks_data = {}
    base_dir = os.path.join(directory, 'merged', rq, merging_method)
    with open(os.path.join(base_dir, 'all_counts.json'), 'r') as f:
        data = json.load(f)
        for item in data:
            tasks = item['file'].split('_')[1]
            merged_tasks_data[tasks] = item

    # get intersections
    intersections = []
    for merged_task in merged_tasks_data.keys():
        intersections.append([])
        constituent_tasks = merged_task.split('-')
        for task in constituent_tasks:
            intersection = get_intersections(merged_tasks_data[merged_task], individual_tasks_data[task], merged_task, task)
            # print(f"Intersection between {merged_task} and {task}: {intersection}")
            intersections[-1].append(intersection)
    
    with open(os.path.join(directory, INTERSECT_DIR, f"{rq}-{merging_method}.json"), 'w') as f:
        json.dump(intersections, f, indent=4)
        
def main():
    # generate_summaries_and_counts()
    # get_correct_intersections()
    # summerize_counts(ROOT_DIR+'/merged/rq-3/weight-averaging')
    # summerize_counts_individual(ROOT_DIR)
    # gather_all_counts_individual(ROOT_DIR)
    gather_all_counts(ROOT_DIR)
    # log_all_intersections(ROOT_DIR, 'rq-2', 'dare_ties')

if __name__ == '__main__':
    main()