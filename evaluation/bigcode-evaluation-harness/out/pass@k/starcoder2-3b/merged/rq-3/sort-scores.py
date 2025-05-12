import os

def sort_scores():
    dirs = os.listdir('.')
    for d in dirs:
        if not os.path.isdir(d):
            continue

        three_tasks_scores = []
        four_tasks_scores = []
        with open(os.path.join(d, 'scores.csv'), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                # print(parts)
                if len(parts) != 3:
                    continue
                # round parts 2 and 3 to 2 decimal places
                parts[1] = round(float(parts[1]) * 100, 2)
                parts[2] = round(float(parts[2]) * 100, 2)
                if len(parts[0].split('-')) == 3:
                    three_tasks_scores.append(parts)
                else:
                    four_tasks_scores.append(parts)
        
        three_tasks_scores.sort(key=lambda x: x[0], reverse=False)
        four_tasks_scores.sort(key=lambda x: x[0], reverse=False)
        scores = three_tasks_scores + four_tasks_scores
        with open(os.path.join(d, 'sorted-scores.csv'), 'w') as f:
            for s in scores:
                f.write(','.join(map(str, s)) + '\n')

def concatenate_scores():
    dirs = os.listdir('.')
    scores = []
    i = 0
    # concatenate the second column of the sorted-scores.csv files horizontally
    for d in dirs:
        if not os.path.isdir(d):
            continue
        print('inserting from: ', d)
        with open(os.path.join(d, 'sorted-scores.csv'), 'r') as f:
            for j, line in enumerate(f):
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue
                if i == 0:
                    scores.append(parts[0] + ',' + parts[1])
                else:
                    scores[j] += ',' + parts[1]
        i += 1

    with open('all-scores.csv', 'w') as f:
        for s in scores:
            f.write(s + '\n')

if __name__ == '__main__':
    # sort_scores()
    concatenate_scores()
