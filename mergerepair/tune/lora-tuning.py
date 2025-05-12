import os
import json
import torch
import math

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)

COFIG_FILE = '../configs/granite_config.json'

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )

def train(conf):
    # config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=conf['lora_rank'],
        target_modules=[
            'q_proj',
            'o_proj',
            'k_proj',
            'v_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
        ],
        task_type='CAUSAL_LM',
    )

    # load task-specific dataset
    root_data_path = conf['data_path']
    task = conf['task']

    # find file name
    file_name = ''
    for file in os.listdir(os.path.join(root_data_path, task)):
        if file.endswith('.json'):
            file_name = file
            break

    data_files = {'train': f'{root_data_path}/{task}/{file_name}'}
    dataset = load_dataset('json', data_files=data_files, split='train')
    
    if conf['do_validation']:
        dataset = dataset.train_test_split(test_size=conf['test_split'], seed=conf['seed'])

    if conf['small_sampling']:
        if conf['do_validation']:
            dataset['train'] = dataset['train'].select(range(100))
            conf['max_steps'] = 10
            dataset['test'] = dataset['test'].select(range(100))
        else:
            dataset = dataset.select(range(100))
            conf['max_steps'] = 10

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        conf['model_path'],
        quantization_config=bnb_config,
        device_map={'': PartialState().process_index},
    )

    print_trainable_parameters(model)
    print(model)

    # based on the instruction format in https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/tasks/humanevalpack.py
    def formatting_instructions(example):
        return f"Question: {example[conf['instruction_field']]}\n{example[conf['input_field']]}\n\nAnswer:\n{example[conf['output_field']]}"

    # set the maximum number of steps based on the dataset size and number of epochs
    if conf['do_validation']:
        max_steps = (len(dataset['train']) // conf['train_micro_batch_size'] // conf['gradient_accumulation_steps']) * conf['epochs']
        eval_strategy = 'steps'
    else:
        max_steps = (len(dataset) // conf['train_micro_batch_size'] // conf['gradient_accumulation_steps']) * conf['epochs']
        eval_strategy = 'no'

    # setup the trainer args
    args = SFTConfig(
        per_device_train_batch_size=conf['train_micro_batch_size'],
        per_device_eval_batch_size=conf['eval_micro_batch_size'],
        gradient_accumulation_steps=conf['gradient_accumulation_steps'],
        do_train=True,
        do_eval=conf['do_validation'],
        evaluation_strategy=eval_strategy,
        packing=True,
        warmup_steps=conf['warmup_steps'],
        max_steps=max_steps,
        learning_rate=conf['learning_rate'],
        lr_scheduler_type=conf['lr_scheduler_type'],
        weight_decay=conf['weight_decay'],
        bf16=conf['bf16'],
        logging_strategy='steps',
        logging_steps=conf['logging_steps'],
        output_dir=os.path.join(conf['output_dir'], conf['model_path'].split('/')[-1], conf['task']),
        optim='paged_adamw_8bit',
        neftune_noise_alpha=conf['neftune_noise_alpha'],
        seed=conf['seed'],
        run_name=str(conf['model_path'].split('/')[-1]) + '-' + str(task.split(' ')[-1]),
        report_to='mlflow',
    )

    # setup the trainer
    if conf['do_validation']:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            max_seq_length=conf['max_seq_length'],
            args=args,
            peft_config=lora_config,
            formatting_func=formatting_instructions,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            max_seq_length=conf['max_seq_length'],
            args=args,
            peft_config=lora_config,
            formatting_func=formatting_instructions,
        )

    # launch training
    print('Training...')
    trainer.train()

    # save the model checkpoint
    print('Saving the last checkpoint of the model')
    model.save_pretrained(os.path.join(conf['output_dir'], conf['model_path'].split('/')[-1], conf['task'], 'final_checkpoint/'))
    print('Training Done!') 

def main():
    # read config file
    conf = json.load(open(COFIG_FILE))

    # set seed
    set_seed(conf['seed'])

    # train!
    train(conf)

if __name__ == '__main__':
    main()