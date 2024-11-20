import os
import argparse
import torch

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    TrainingArguments,
    Trainer,
)
import datasets

from data_utils import LlavaNextDataCollator

def formatting_func(examples):
    messages = []
    
    for inst,res in zip(examples['instruction'],examples['response']):
        message = [
            {'role': 'user', 'content': f"<image>\n{inst}"},
            {'role': 'assistant', 'content': f"{res}"},
        ]
        messages.append(message)
        
    return {
        'messages': messages
    }

if __name__=='__main__':
    local_rank=os.environ['LOCAL_RANK']    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        'MLP-VLM/llava-next-bllossom-8b-base',
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
    )
    
    processor = LlavaNextProcessor.from_pretrained('MLP-VLM/llava-next-bllossom-8b-base')
    
    for n,p in model.named_parameters():
        if 'language_model' in n:
            p.requires_grad=False
    
    if local_rank == 0:
        for n,p in model.named_parameters():
                if p.requires_grad:
                    print(n)
    
    dataset = datasets.load_dataset('MLP-VLM/bllossom-vision','aihub-chart-table',split='train')
    # chart = datasets.load_dataset('MLP-VLM/bllossom-vision','aihub-chart',split='train')
    # table = datasets.load_dataset('MLP-VLM/bllossom-vision','aihub-table',split='train')
    # dataset = datasets.concatenate_datasets([chart,table]).shuffle()
    
    # dataset = dataset.map(
    #     formatting_func,
    #     num_proc=128,
    #     batched=True
    # )
    
    training_args = TrainingArguments(
        output_dir='/home/work/hdd_data/VLM/llava-next/results/doc-stage1',
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        lr_scheduler_type='cosine',
        warmup_ratio=0.01,
        logging_strategy='steps',
        logging_steps=1,
        save_strategy='epoch',
        save_total_limit=3,
        bf16=True,
        optim='adamw_bnb_8bit',
        dataloader_num_workers=32,
        remove_unused_columns=False,
        label_names=['labels'],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        report_to='wandb',
        run_name='llava-next-stage1'
    )
    
    collator = LlavaNextDataCollator(processor)
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=collator,
    )
    
    trainer.train()
