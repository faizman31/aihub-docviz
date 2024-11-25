import evaluate
import jsonlines as jsonl
import glob
from transformers import AutoTokenizer
import datasets
import json
import os

def calculate_recall(f1_score, precision):
    if precision == 0:
        raise ValueError("Precision cannot be zero.")
    recall = (2 * precision * f1_score) / (precision + f1_score)
    return recall
    
hf_token = os.environ['HF_TOKEN']

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
file_list = glob.glob('/aihub-docviz/results/*.jsonl')
examples = []

for file_name in file_list:
    with jsonl.open(file_name,'r') as f:
        for line in f:
            examples.append(line)


tokenizer = AutoTokenizer.from_pretrained('MLP-VLM/mllama_3b_stage2_base_10epoch',token=hf_token)
sample_set = {'predictions':[example['predict'].strip() for example in examples if not example['label'].startswith('###')],'references':[example['label'].strip() for example in examples if not example['label'].startswith('###')]}
rouge_score = rouge.compute(predictions=sample_set['predictions'], references=sample_set['references'],tokenizer=lambda x: tokenizer.tokenize(x))
bleu_score = bleu.compute(predictions=sample_set['predictions'], references=sample_set['references'],tokenizer=lambda x: tokenizer.tokenize(x))

score_result = {
    'rouge-1':{
                'f1': rouge_score['rouge1'],
                'precision': bleu_score['precisions'][0],
                'recall': calculate_recall(rouge_score['rouge1'],bleu_score['precisions'][0])
              },
    'rouge-L':{
                'f1': rouge_score['rouge1'],
                'precision': bleu_score['precisions'][2],
                'recall': calculate_recall(rouge_score['rougeL'],bleu_score['precisions'][2])
              }
},

with open('/aihub-docviz/results/evaluation_result.json','w') as f:
    json.dump(score_result,f,indent=4,ensure_ascii=False)
