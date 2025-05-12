import scipy.stats as stats
import os
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from effect_size_analysis.cliff_delta import cliff_delta

MODEL = 'starcoder2-3b'
method = 'weight-averaging'
rq = 'rq-2'
PATH = f'correct path to bigcode-evaluation-harness/out/{MODEL}/merged/{rq}/{method}'
# for pass10
# BASE = f'correct path to bigcode-evaluation-harness/out/{MODEL}/Program Repair/logs_summerized.json'
# for pass1
BASE = f'correct path to bigcode-evaluation-harness/out/{MODEL}/Program Repair/logs_summerized_1.json'
SAMPLING_TIMES = 100

def base_vs_others_statistical_tests():
    others_probs = {}
    base_probs = []
    mean_samples = []
    mean_base_samples = []

    with open(BASE) as f:
        data = json.load(f)
        
        for problem_info in data:
            base_probs.append(problem_info['prob'])
                    
        base_samples = [np.random.choice(base_probs, len(base_probs), replace=True) for _ in range(SAMPLING_TIMES)]
        for sample in base_samples:
            mean_base_samples.append(np.mean(sample))
        
        assert len(mean_base_samples) == SAMPLING_TIMES

    ttests = []
    wilcoxon = []
    cliffs = []
    
    for file in os.listdir(PATH):
        if file.__contains__('summerized1'):
            if file.__contains__(BASE):
                continue
            
            else:
                others_probs[file] = []
                with open(os.path.join(PATH, file)) as f:
                    data = json.load(f)
                
                probs = []
                for problem_info in data:
                    probs.append(problem_info['prob'])
                    # others_probs[file].append(problem_info['prob'])

                samples = [np.random.choice(probs, len(probs), replace=True) for _ in range(SAMPLING_TIMES)]
                for sample in samples:
                    mean_samples.append(np.mean(sample))

                assert len(mean_samples) == SAMPLING_TIMES

                # print(mean_samples[:50])
                # print(mean_base_samples[:50])

                ttest_p = stats.ttest_ind(mean_samples, mean_base_samples)[1]
                wilcoxon_p = stats.wilcoxon(mean_samples, mean_base_samples)[1]
                cliffs_p = cliff_delta(s1=mean_samples, s2=mean_base_samples, alpha=0.05, accurate_ci=True)[0]

                ttests.append((file.split('_')[1], ttest_p))
                wilcoxon.append((file.split('_')[1], wilcoxon_p))
                cliffs.append((file.split('_')[1], cliffs_p))
    
                mean_samples = []

    ttests = sorted(ttests, key=lambda x: x[0])
    wilcoxon = sorted(wilcoxon, key=lambda x: x[0])
    cliffs = sorted(cliffs, key=lambda x: x[0])

    print("Statistical tests:")
    print('t-test')
    print(ttests)
    print('wilcoxon')
    print(wilcoxon)
    print('cliff_delta')    
    print(cliffs)

    with open(os.path.join(PATH, f'statistical_tests_1.csv'), 'w') as f:
        f.write('tasks,ttest_p,wilcoxon_p,cliff_delta\n')
        for i in range(len(ttests)):
            f.write(f'{ttests[i][0]},{ttests[i][1]},{wilcoxon[i][1]},{cliffs[i][1]}\n')
    
    print("Statistical tests have been saved successfully.")

def effect_size_initial_data():
    base_probs = []

    with open(BASE) as f:
        data = json.load(f)
        
        for problem_info in data:
            base_probs.append(problem_info['prob'])
    
    for file in os.listdir(PATH):
        if file.__contains__('summerized1'):
            if file.__contains__(BASE):
                continue
            
            else:
                with open(os.path.join(PATH, file)) as f:
                    data = json.load(f)
                
                probs = []
                for problem_info in data:
                    probs.append(problem_info['prob'])
                
                print(f"Cliff's Delta of {file.split('_')[1]} vs base: ", cliff_delta(s1=base_probs, s2=probs, alpha=0.05, accurate_ci=True))
                print()

if __name__ == '__main__':
    base_vs_others_statistical_tests()
    # effect_size_initial_data()