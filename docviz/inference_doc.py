from PIL import Image
import requests
from tqdm.auto import tqdm
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor
)
from peft import PeftModel
import torch
import json
from datasets import load_dataset,concatenate_datasets
import argparse 
from datetime import datetime    

def parse_argument():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--num_gpus', type=int, default=8)
    
    args = parser.parse_args()
    
    return args

def write_to_file(data, file_path):
    with open(file_path, 'a') as file:
        json.dump(data, file,ensure_ascii=False)
        file.write('\n')

def formatting_func(examples):
    messages = []

    for inst,res in zip(examples['instruction'],examples['response']):
        message = [
            {'role':'user', 'content': f"<image>\n{inst}"},
            {'role':'assistant', 'content':f"{res}"},
        ]
        messages.append(message)
    
    return {
        'messages':messages
    }

args = parse_argument()

model = LlavaNextForConditionalGeneration.from_pretrained(
    'ShinDJ/llava-next-docviz',
    attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16,
).to(f'cuda')

processor = LlavaNextProcessor.from_pretrained('ShinDJ/llava-next-docviz')

model.eval()

test_dataset = load_dataset('MLP-VLM/bllossom-vision','aihub-docviz',split='test')

test_dataset = test_dataset.map(
    formatting_func,
    num_proc=32,
    batched=True,
    remove_columns=['instruction','response']
)

results = []

chunk_size = len(test_dataset) / args.num_gpus
start = round(chunk_size * (args.gpu_id))
end = round(chunk_size * (args.gpu_id + 1))

save_path = f'../results/result_vlm_{args.gpu_id}.jsonl'

for idx in tqdm(range(start, end)):
    image = test_dataset[idx]['images']
    instruction = test_dataset[idx]['messages'][0]['content']
    input_text = processor.tokenizer.apply_chat_template(
        test_dataset[idx]['messages'][:1],
        tokenize=False,
        add_generation_prompt=True,
    )
    label = test_dataset[idx]['messages'][-1]['content']
        
    inputs = processor(
        text=input_text,
        images=image,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=[processor.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
    )
    result = processor.batch_decode(output)[0]
    start_idx = result.find('<|start_header_id|>assistant<|end_header_id|>\n\n')
    predict = result[start_idx:].replace('<|start_header_id|>assistant<|end_header_id|>','').replace('<|eot_id|>','').strip()
    
    now = datetime.now()
    data_id = test_dataset[idx]['qa_id']
    data = {'time': now.strftime('%Y/%m/%d %H:%M:%S'), 'index':data_id,'instruction':instruction, 'predict':predict, 'label':label}
    write_to_file(
        data, 
        save_path,
    )   
    
