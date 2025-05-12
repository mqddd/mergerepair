import scipy.stats as stats
import os

MODEL = 'starcoder2-3b'
ROOT_DIR = f'correct path to bigcode-evaluation-harness/out/{MODEL}/merged'
RQS = ['rq-1', 'rq-2', 'rq-3']
MERGING_METHODS = ['weight-averaging', 'ties', 'dare_ties']
BASE_SCORES = {'pass1': 28.66, 'pass10': 39.11}

def one_sample_t_test():
    # Load the data
    for rq in RQS:
        rq_scores = {}
        for method in MERGING_METHODS:
            print(f'{rq} - {method}')
            with open(os.path.join(ROOT_DIR, rq, method, 'scores.csv'), 'r') as f:
                data = f.readlines()

            data = [line.strip().split(',') for line in data]
            pass1 = [round(float(line[1]) * 100, 2) for line in data]
            pass10 = [round(float(line[2]) * 100, 2) for line in data]

            # print('pass1: ', pass1)
            # print('pass10', pass10)

            base_pass1 = BASE_SCORES['pass1']
            base_pass10 = BASE_SCORES['pass10']

            # perform one-sample t-test
            t1, p1 = stats.ttest_1samp(pass1, base_pass1)
            # print(f'Pass 1 - t: {t}, p: {p}')

            t10, p10 = stats.ttest_1samp(pass10, base_pass10)
            # print(f'Pass 10 - t: {t}, p: {p}')

            rq_scores[method] = {'pass1': {'t': t1, 'p': p1}, 
                                 'pass10': {'t': t10, 'p': p10}}
            
        # save the results in a file
        with open(os.path.join(ROOT_DIR, rq, 'statistical-tests.csv'), 'w') as f:
            f.write('method,pass1_t,pass1_p,pass10_t,pass10_p\n')
            for method in rq_scores:
                f.write(f'{method},{rq_scores[method]["pass1"]["t"]},{rq_scores[method]["pass1"]["p"]},{rq_scores[method]["pass10"]["t"]},{rq_scores[method]["pass10"]["p"]}\n')
        

def two_distribution_t_test():
    # Load the data
    for rq in RQS:
        rq_scores = {}
        for method in MERGING_METHODS:
            print(f'{rq} - {method}')
            with open(os.path.join(ROOT_DIR, rq, method, 'scores.csv'), 'r') as f:
                data = f.readlines()

            data = [line.strip().split(',') for line in data]
            pass1 = [round(float(line[1]) * 100, 2) for line in data]
            pass10 = [round(float(line[2]) * 100, 2) for line in data]

            rq_scores[method] = {
                'pass1': pass1,
                'pass10': pass10
            }

    

    # calculate the t-test for the pair of score methods
    for rq in RQS:
        scores = {}
        for method1 in MERGING_METHODS:
            for method2 in MERGING_METHODS:
                if method1 == method2:
                    continue
                print(f'{rq} - {method1} vs {method2}')
                t1, p1 = stats.ttest_ind(rq_scores[method1]['pass1'], rq_scores[method2]['pass1'])
                t10, p10 = stats.ttest_ind(rq_scores[method1]['pass10'], rq_scores[method2]['pass10'])

                scores[(method1, method2)] = {'pass1': {'t': t1, 'p': p1},
                                                 'pass10': {'t': t10, 'p': p10}}
        
        # save the results in a file
        with open(os.path.join(ROOT_DIR, rq, 'statistical-tests.csv'), 'w') as f:
            f.write('method1,method2,pass1_t,pass1_p,pass10_t,pass10_p\n')
            for methods in scores:
                f.write(f'{methods[0]},{methods[1]},{scores[methods]["pass1"]["t"]},{scores[methods]["pass1"]["p"]},{scores[methods]["pass10"]["t"]},{scores[methods]["pass10"]["p"]}\n')

if __name__ == '__main__':
    # one_sample_t_test()
    two_distribution_t_test()