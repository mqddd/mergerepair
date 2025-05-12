import os
import json
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.manifold import TSNE

ROOT_DIR = 'correct path to bigcode-evaluation-harness/out/starcoder2-3b/'
INDIVIDUAL_TASKS = ['Development', 'Improvement', 'Misc', 'Program Repair', 'Test & QA']
MERGED = 'merged'

RQS = ['rq-1', 'rq-2', 'rq-3']
MERGING_METHODS = ['weight-averaging', 'ties', 'dare_ties']
MODEL_PATH = 'correct path to models/starcoder2-3b'

def get_merged_projections(directory):
    # getting merged tasks embeddings and projections
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    for rq in RQS:
        for method in MERGING_METHODS:
            all_files = os.listdir(os.path.join(directory, MERGED, rq, method))
            projections_dict = {}
            for i in tqdm(range(len(all_files))):
                file = all_files[i]
                if file.endswith('.json') and file.startswith('generations'):
                    with open(os.path.join(directory, MERGED, rq, method, file), 'r') as f:
                        data = json.load(f)
                        # get the task attribute
                        tokenized_sols = []
                        for problem_sols in data:
                            counter = 0  
                            # for sol in problem_sols:
                            #     if counter > 0:
                            #         continue
                            tokenized_sol = tokenizer.encode(problem_sols[0])
                                # counter += 1
                            
                            # control the length here
                            tokenized_sols.append(tokenized_sol)
                            # print(len(tokenized_sols))
                            # print(len(tokenized_sols[-1]))

                    # find max and min
                    max = -1
                    min = 100000
                    for sol in tokenized_sols:
                        if len(sol) > max:
                            max = len(sol)
                        if len(sol) < min:
                            min = len(sol)

                    print(max, min)

                    break

                    # convert the tokenized solutions to numpy array
                    tokenized_sols = np.array(tokenized_sols)
                    # print(tokenized_sols.shape)

                    # plot the features
                    # there is a problem here!
                    a = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
                    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
                    projections = tsne.fit_transform(tokenized_sols)
                    # print(projections)
                    projections_dict[file.split('_')[-1][:-5]] = projections

            # scats = []
            # for key, value in projections_dict.items():
            #     scats.append(go.Scatter(
            #         # value, 
            #         x=value[:, 0],
            #         y=value[:, 1],
            #         name=key,
            #         mode='markers',
            #         # z=2,
            #         # color=df.species, 
            #         # labels={'color': 'species'}
            #     ))
            # # fig.update_traces(marker_size=1)
            # fig = go.Figure(data=scats)
            # fig.show()
            # print(f"projections_dict of {key} added")
            # print('-------------------')
    
    return projections_dict

def get_individual_projections(directory):
    # getting individual tasks embeddings and projections
    current_dirs = os.listdir(directory)
    for task in INDIVIDUAL_TASKS:
        if task not in current_dirs:
            continue
        all_files = os.listdir(os.path.join(directory, task))
        projections_dict = {}
        for i in tqdm(range(len(all_files))):
            file = all_files[i]
            if file.endswith('.json') and file.startswith('generations'):
                with open(os.path.join(directory, task, file), 'r') as f:
                    data = json.load(f)
                    tokenized_sols = []
                    for problem_sols in data:
                        counter = 0  

                        tokenized_sol = tokenizer.encode(problem_sols[0])
                                    
                        tokenized_sols.append(tokenized_sol[:10])

                # convert the tokenized solutions to numpy array
                tokenized_sols = np.array(tokenized_sols)
                # print(tokenized_sols.shape)

                # plot the features
                # there is a problem here!
                a = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
                tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)
                projections = tsne.fit_transform(tokenized_sols)
                # print(projections)
                projections_dict[file.split('_')[-1][:-5]] = projections

        # scats = []
        # for key, value in projections_dict.items():
        #     scats.append(go.Scatter(
        #         # value, 
        #         x=value[:, 0],
        #         y=value
        #     ))
                
    return projections_dict


if __name__ == '__main__':
    merged_projections = get_merged_projections(ROOT_DIR)
    # individual_projections = get_individual_projections(ROOT_DIR)