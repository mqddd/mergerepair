import os
import itertools

from enum import Enum
from tqdm import tqdm
from transformers import (
    set_seed,
)
from copy import deepcopy
from pathlib import Path
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_PATH = 'correct path to models/starcoder2-3b'
MODEL_NAME = 'starcoder2-3b'
BASE_TASK_PATH = os.path.join('correct path to mergerepair/out', MODEL_NAME)
TASKS = {'T1': 'Program Repair', 
         'T2': 'Improvement',
         'T3': 'Misc', 
         'T4': 'Development', 
         'T5': 'Test & QA'}

CONTINUAL_TASKS = {'T1': 'Program Repair', 
         'T2': 'Improvement',
         'T4': 'Development', 
         'T5': 'Test & QA'}

class MergingMethod(Enum):
    AVERAGE = 'weight-averaging'
    TIES = 'ties'
    DARE = 'dare_ties'

def sorted_list_dir(directory):
    def get_creation_time(item):
        return item.stat().st_ctime

    path_object = Path(directory)
    items = path_object.iterdir()
    sorted_items = sorted(items, key=get_creation_time)
    return [item.name for item in sorted_items]

# RQ1 and RQ2: improvement on and generalizability to APR
def equal_weight_merging(rq='rq1', merging_method=MergingMethod.AVERAGE):    
    # merge different combinations of adapters
    for i in tqdm(range(2, len(TASKS) + 1)):
        for subset in itertools.combinations(TASKS.keys(), i):
            if merging_method == MergingMethod.AVERAGE:
                if rq == 'rq1' and 'T1' not in subset:
                    continue
                if rq == 'rq2' and 'T1' in subset:
                    continue
                # if subset != 'T1-T2-T3':
                #     continue
                # load the adapters
                loaded_adapters = {}
                checkpoints_epochs = {}
                print('merging task-specific-adapters using weight averaging:', subset)
                for task in subset:
                    # TODO: check if the adapter path is correct
                    task_path = os.path.join(BASE_TASK_PATH, TASKS[task])
                    checkpoints = sorted_list_dir(task_path)
                    loaded_adapters[task] = load_file(os.path.join(task_path, checkpoints[-2], 'adapter_model.safetensors'))
                    checkpoints_epochs[task] = checkpoints[-2].split('-')[-1]
                    print(f"Loaded adapter for {task} from {os.path.join(task_path, checkpoints[-2], 'adapter_model.safetensors')}")

                # average the loaded adapters
                summed_adapter = deepcopy(list(loaded_adapters.values())[0])
                for task, adapter in list(loaded_adapters.items())[1:]:
                    for l1, l2 in zip(summed_adapter, adapter):
                        summed_adapter[l1] += adapter[l2]
                        
                merged_adapter = {layer: tensor / len(loaded_adapters) for layer, tensor in summed_adapter.items()}
        
                # save the merged adapter
                try:
                    merged_names = '-'.join([task for task in subset])
                    epochs = '-'.join([checkpoints_epochs[task] for task in subset])
                    os.makedirs(os.path.join(BASE_TASK_PATH, f'merged-{rq}', f'{merging_method.value}', f'{merged_names}'), exist_ok=True)
                    save_file(merged_adapter, os.path.join(BASE_TASK_PATH, f'merged-{rq}', 
                                                           f'{merging_method.value}', 
                                                           merged_names, 
                                                           'adapter_model.safetensors'))
                    print(f'Merged adapter for {subset} using {merging_method.value} saved successfully!')
                except Exception as e:
                    print(f'Error while saving the merged adapter for {subset}: {e}')

            elif merging_method == MergingMethod.TIES or merging_method == MergingMethod.DARE:
                if rq == 'rq1' and 'T1' not in subset:
                    continue
                if rq == 'rq2' and 'T1' in subset:
                    continue
                # if subset != 'T1-T2-T3':
                #     continue
                loaded_adapters = []
                model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
                counter = 0
                for task in subset:
                    adapter_task_path = os.path.join(BASE_TASK_PATH, TASKS[task])
                    checkpoints = sorted_list_dir(adapter_task_path)
                    if counter == 0:
                        model = PeftModel.from_pretrained(model, os.path.join(adapter_task_path, checkpoints[-2]), adapter_name=task)
                        loaded_adapters.append(task)
                        counter += 1
                    else:
                        _ = model.load_adapter(os.path.join(adapter_task_path, checkpoints[-2]), adapter_name=task)
                        loaded_adapters.append(task)
                        counter += 1
                
                # merge the adapters
                density = 1
                if merging_method == MergingMethod.DARE:
                    # density of dare_ties approach
                    density = 0.8
                model.add_weighted_adapter(loaded_adapters, 
                                            weights=[1.0]*len(loaded_adapters),
                                            adapter_name='-'.join(loaded_adapters), 
                                            combination_type=merging_method.value,
                                            density=density)
            
                # save the merged adapter
                try:
                    save_path = os.path.join(BASE_TASK_PATH, 'merged-' + rq, merging_method.value)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    model.save_pretrained(save_path, selected_adapters=['-'.join(loaded_adapters)])
                    print(f'Merged adapter for {subset} using {merging_method.value} saved successfully!')
                except Exception as e:
                    print(f'Error while saving the merged adapter for {subset}: {e}')
 
            else:
                raise ValueError('Invalid merging method!')

# RQ3: continual merging of adapters
def continual_merging(merging_method=MergingMethod.AVERAGE):    
    for i in tqdm(range(3, len(CONTINUAL_TASKS) + 1)):
        for subset in itertools.permutations(CONTINUAL_TASKS.keys(), i):
            if 'T1' not in subset:
                continue
            if merging_method == MergingMethod.AVERAGE:
                # load the adapters
                loaded_adapters = {}
                checkpoints_epochs = {}
                print('merging task-specific adapters using weight averaging:', subset)
                for task in subset:
                    # TODO: check if the adapter path is correct
                    adapter_task_path = os.path.join(BASE_TASK_PATH, TASKS[task])
                    checkpoints = sorted_list_dir(adapter_task_path)
                    loaded_adapters[task] = load_file(os.path.join(adapter_task_path, checkpoints[-2], 'adapter_model.safetensors'))
                    checkpoints_epochs[task] = checkpoints[-2].split('-')[-1]
                    print(f"Loaded adapter for {task} from {os.path.join(adapter_task_path, checkpoints[-2], 'adapter_model.safetensors')}")

                # continual merging the loaded adapters using weight averaging
                merged_adapter = deepcopy(list(loaded_adapters.values())[0])
                for task, adapter in list(loaded_adapters.items())[1:]:
                    for l1, l2 in zip(merged_adapter, adapter):
                        merged_adapter[l1] += adapter[l2]
                        merged_adapter[l1] /= 2
                
                # save the merged adapter
                try:
                    merged_names = '-'.join([task for task in subset])
                    # create final directory
                    os.makedirs(os.path.join(BASE_TASK_PATH, 'merged-rq3', f'{merging_method.value}', f'{merged_names}'), exist_ok=True)
                    save_file(merged_adapter, os.path.join(BASE_TASK_PATH, 
                                                           'merged-rq3',
                                                           f'{merging_method.value}', 
                                                           f'{merged_names}',
                                                           'adapter_model.safetensors'))
                    print(f'Merged adapter for {subset} using {merging_method.value} saved successfully!')
                except Exception as e:
                    print(f'Error while saving the merged adapter for {subset}: {e}')
            elif merging_method == MergingMethod.TIES or merging_method == MergingMethod.DARE:
                last_loaded_adapter = None
                previous_merged_adapter = None
                merged_adapter = None
                model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
                counter = 0
                density = 1
                if merging_method == MergingMethod.DARE:
                    # density of dare_ties approach
                    density = 0.8
                for task in subset:
                    adapter_task_path = os.path.join(BASE_TASK_PATH, TASKS[task])
                    checkpoints = sorted_list_dir(adapter_task_path)
                    if counter == 0:
                        model = PeftModel.from_pretrained(model, os.path.join(adapter_task_path, checkpoints[-2]), adapter_name=task)
                        last_loaded_adapter = task
                        merged_adapter = task
                        counter += 1
                    else:
                        _ = model.load_adapter(os.path.join(adapter_task_path, checkpoints[-2]), adapter_name=task)
                        last_loaded_adapter = task
                        previous_merged_adapter = merged_adapter
                        merged_adapter += f'-{task}'

                        print('merged adapter: ', merged_adapter, ' from subset: ', subset)

                        model.add_weighted_adapter([last_loaded_adapter, previous_merged_adapter], 
                                                   weights=[1.0, 1.0],
                                                   adapter_name=merged_adapter, 
                                                   combination_type=merging_method.value,
                                                   density=density)
                        
                        # only keeping the last merged adapter, delete the previous ones
                        model.delete_adapter(last_loaded_adapter)
                        model.delete_adapter(previous_merged_adapter)
                        counter += 1
                
                # save the merged adapter
                try:
                    save_path = os.path.join(BASE_TASK_PATH, 'merged-rq3', merging_method.value)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    model.save_pretrained(save_path, selected_adapters=[merged_adapter])
                    print(f'Merged adapter for {merged_adapter} using {merging_method.value} saved successfully!')
                except Exception as e:
                    print(f'Error while saving the merged adapter for {merged_adapter}: {e}')

            else: 
                raise ValueError('Invalid merging method!')

def test_merged_adapters():
    # test the merged adapters
    path_1 = 'correct path to mergerepair/out/starcoder2-3b/merged-rq1/dare_ties/T1-T2-T4/adapter_model.safetensors'
    path_2 = 'correct path to mergerepair/out/starcoder2-3b/merged-rq1/ties/T1-T2-T4/adapter_model.safetensors'

    adapter1 = load_file(path_1)
    adapter2 = load_file(path_2)

    # check if the adapters are the same
    for l1, l2 in zip(adapter1, adapter2):
        print(l1, l2)
        print(adapter1[l1] == adapter2[l2])


def main():
    # set seed
    set_seed(42)

    # RQ1 - improvement on APR and RQ2 - generalizability to APR
    # equal_weight_merging(rq='rq2', merging_method=MergingMethod.AVERAGE)

    # RQ3 - continual merging!
    continual_merging(merging_method=MergingMethod.DARE)

    # testing equality of merged adapters
    # test_merged_adapters()
        

if __name__ == '__main__':
    main()