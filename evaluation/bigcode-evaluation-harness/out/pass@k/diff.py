import os

ROOT_DIR = 'correct path to bigcode-evaluation-harness/out/starcoder2-3b/merged/rq-3/'
METHODS = ['dare_ties', 'ties', 'weight-averaging']

def main():
    # read csv file from the root directory
    tasks = {}
    for meth in METHODS:
        lines = []
        file = os.path.join(ROOT_DIR, meth, 'scores.csv')
        with open(file, 'r') as f:
            while True:
                line = f.readline()
                lines.append(line.split(',')[0])
                if not line:
                    break

        tasks[meth] = lines

    # get the different tasks    
    tasks1 = tasks['dare_ties']
    tasks2 = tasks['ties']
    tasks3 = tasks['weight-averaging']

    # get the different tasks among the three methods
    diff1 = set(tasks3) - set(tasks2)
    diff2 = set(tasks3) - set(tasks1)

    print('diff of weight-averaging and ties:', diff1)
    print('diff of ties and dare_ties:', diff2)   
        
if __name__ == '__main__':
    main()