import os
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np 

ROOT_DIR = 'correct path to bigcode-evaluation-harness/out/starcoder2-3b/merged/'
RQS = ['rq-1', 'rq-2', 'rq-3']
MERGING_METHODS = ['weight-averaging', 'ties', 'dare_ties']
EVAL_TASK = 'humanevalfixtests-python'
COLORS = {'green': '#00796B', 'purple': '#7C4DFF', 'orange': '#FF9800'}

TASKS = {'T1': 'Program Repair', 
         'T2': 'Improvement',
         'T3': 'Misc', 
         'T4': 'Development', 
         'T5': 'Test & QA'}

# RQ 1 tasks
RQ1_TASKS=("T1-T2" "T1-T3" "T1-T4" "T1-T5" "T1-T2-T3" "T1-T2-T4" "T1-T2-T5" "T1-T3-T4" "T1-T3-T5" "T1-T4-T5" "T1-T2-T3-T4" "T1-T2-T3-T5" "T1-T2-T4-T5" "T1-T3-T4-T5" "T1-T2-T3-T4-T5")

# RQ 2 tasks
RQ2_TASKS=("T2-T3" "T2-T4" "T2-T5" "T3-T4" "T3-T5" "T4-T5" "T2-T3-T4" "T2-T3-T5" "T2-T4-T5" "T3-T4-T5" "T2-T3-T4-T5")

# RQ 3 tasks
RQ3_TASKS=('T1-T4-T2-T5' 'T1-T2-T4' 'T5-T1-T4' 'T2-T1-T5-T4' 'T5-T2-T1' 'T4-T5-T1-T2' 'T5-T1-T4-T2' 'T2-T1-T4-T5' 'T4-T1-T5' 'T1-T2-T4-T5' 
'T5-T1-T2-T4' 'T2-T5-T1' 'T4-T2-T1-T5' 'T2-T1-T4' 'T5-T4-T1' 'T4-T1-T2-T5' 'T1-T5-T4-T2' 'T5-T4-T1-T2' 'T4-T5-T1' 'T5-T1-T2' 'T4-T1-T2' 
'T1-T5-T4' 'T4-T1-T5-T2' 'T2-T4-T5-T1' 'T1-T5-T2-T4' 'T1-T4-T5-T2' 'T1-T4-T5' 'T4-T2-T5-T1' 'T2-T1-T5' 'T5-T2-T1-T4' 'T2-T4-T1-T5' 'T2-T4-T1' 
'T2-T5-T4-T1' 'T5-T2-T4-T1' 'T4-T2-T1' 'T1-T2-T5' 'T1-T5-T2' 'T1-T2-T5-T4' 'T5-T4-T2-T1' 'T4-T5-T2-T1' 'T1-T4-T2' 'T2-T5-T1-T4')

def retrieve_scores(all_files, merging_path):
    tasks_scores = []
    for file in all_files:
        if file.endswith('.json'):
            with open(os.path.join(merging_path, file), 'r') as f:
                data = json.load(f)
                # get the task attribute
                task = data[EVAL_TASK]
                tasks_scores.append([data['config']['peft_model'].split('/')[-1], 
                                        round(task['pass@1'] * 100, 2), 
                                        round(task['pass@10'] * 100, 2)
                                    ])
    return tasks_scores

def plot_individual_rqs(all_scores, rq='rq-1', merging_method='weight-averaging'):
    bar_x = []
    bar_y = []
    for r, m in all_scores.keys():
        if r != rq or m != merging_method:
            continue
        print(f'{r} and {m}')
        for record in all_scores[(r, m)]:
            bar_x.append(record[0])
            bar_y.append(record[2])
        
    fig = px.bar(x=bar_x, y=bar_y, title=f'{rq} and {merging_method}') 
    horizontal_lines = {
        'T2': 37.99,
        'T4': 32.65,
        'T1': 39.10,
        'T5': 28.90,
        'T3': 41.87
    } 
    for key, value in horizontal_lines.items():
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(bar_x)-0.5,  # Full width of the x-axis
            y0=value, y1=value,  # Constant y value for each horizontal line
            line=dict(color="red", width=2, dash="dash"),
            label=dict(
                text=key,
                textposition='end',
                font=dict(size=15, color='blue'),
                yanchor='top',
            ),
        )

    fig.update_layout(margin=dict(l=100))
    fig.show()

def plot_all_rqs(all_scores):
    # here we define an alias for each task to be compared with the tasks with the same alias
    scores_with_alias= []
    bars = []
    for rq, merging_method in all_scores.keys():
        bar_name = f'{rq}-{merging_method}'
        bar_marker = dict(color=COLORS['green']) if rq == 'rq-1' else dict(color=COLORS['purple']) if rq == 'rq-2' else dict(color=COLORS['orange'])
        bar_marker_2 = dict(color=COLORS['green']) if merging_method == 'weight-averaging' else dict(color=COLORS['purple']) if merging_method == 'ties' else dict(color=COLORS['orange'])

        bar_x = []
        bar_y = []
        for record in all_scores[(rq, merging_method)]:
            if rq == 'rq-1' or rq == 'rq-2':
                # task = record[0]
                # alias = task
                # scores_with_alias.append([alias, record[0], record[1], record[2]])
                # bar_x.append(alias)
                # bar_y.append(record[2])
                pass
            else:
                task = record[0]
                alias = '-'.join(sorted(task.split('-')))
                scores_with_alias.append([alias, record[0], record[1], record[2]])
                bar_x.append(task)
                bar_y.append(record[2])
                pass

        if len(bar_x) == 0 or len(bar_y) == 0:
            continue
        # print('barx', bar_x, 'bary', bar_y)
        bars.append(go.Bar(x=bar_x, y=bar_y, orientation='v', name=bar_name, marker=bar_marker_2))

    fig = go.Figure(data=bars)
    fig.update_layout(barmode='group')
    fig.show()

if __name__ == '__main__':
    all_scores = {}
    for rq in RQS:
        rq_path = os.path.join(ROOT_DIR, rq)
        for merging_method in MERGING_METHODS:
            merging_path = os.path.join(rq_path, merging_method)

            # load all files in the root directory
            all_files = os.listdir(merging_path)

            # select only evaluation files
            all_files = [f for f in all_files if f.startswith('eval')]

            # print(f'{rq} and {merging_method}: collecting from {merging_path} ...!')

            all_scores[(f'{rq}', f'{merging_method}')] = retrieve_scores(all_files, merging_path)
        
    # plot_all_rqs(all_scores)
    plot_individual_rqs(all_scores, 'rq-1', 'weight-averaging')