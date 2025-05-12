import os

ROOT_DIR = 'correct path to bigcode-evaluation-harness/out/starcoder2-3b/merged'
RQS = ['rq-1', 'rq-2', 'rq-3']
MERGING_METHODS = ['weight-averaging', 'ties', 'dare_ties']

def rename_files():
    for rq in RQS:
        for method in MERGING_METHODS:
            all_files = os.listdir(os.path.join(ROOT_DIR, rq, method))
            for file in all_files:
                if '$' in file:
                    os.rename(os.path.join(ROOT_DIR, rq, method, file), os.path.join(ROOT_DIR, rq, method, file.replace('$', '')))

def delete_files():
    for rq in RQS:
        for method in MERGING_METHODS:
            all_files = os.listdir(os.path.join(ROOT_DIR, rq, method))
            for file in all_files:
                if 'summerized' in file or file == 'all_counts.json':
                    os.remove(os.path.join(ROOT_DIR, rq, method, file))

if __name__ == '__main__':
    # rename_files()
    delete_files()