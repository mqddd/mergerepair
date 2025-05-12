import os

ROOT = 'correct path to bigcode-evaluation-harness/out/starcoder2-3b/merged'
METHODS = ['dare_ties', 'ties', 'weight-averaging']
RQS = ['rq-1', 'rq-2', 'rq-3']

if __name__ == '__main__':
    for rq in RQS:
        for meth in METHODS:
            print(f'{rq} - {meth}')
            with open(os.path.join(ROOT, rq, meth, 'scores.csv'), 'r') as f:
                data = f.readlines()

            data = [line.strip().split(',') for line in data]
            pass1 = [(line[0], round(float(line[1]) * 100, 2)) for line in data]
            pass10 = [(line[0], round(float(line[2]) * 100, 2)) for line in data]

            base_pass1 = 28.66
            base_pass10 = 39.11

            # find the difference
            diff_pass1 = [(p[0], round(p[1] - base_pass1, 2)) for p in pass1]
            diff_pass10 = [(p[0], round(p[1] - base_pass10, 2)) for p in pass10]

            # save the difference in a file
            with open(os.path.join(ROOT, rq, meth, 'diff_pass1.csv'), 'w') as f:
                for p in diff_pass1:
                    f.write(f'{p[0]}, {p[1]}\n')
            
            with open(os.path.join(ROOT, rq, meth, 'diff_pass10.csv'), 'w') as f:
                for p in diff_pass10:
                    f.write(f'{p[0]}, {p[1]}\n')