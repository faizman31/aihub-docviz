import evaluate
import jsonlines as jsonl
import glob
from transformers import AutoTokenizer
import datasets
import json

rouge = evaluate.load('rouge')
file_list = glob.glob('/aihub-docviz/results/*.jsonl')
examples = []

for file_name in file_list:
    with jsonl.open(file_name,'r') as f:
        for line in f:
            examples.append(line)


tokenizer = AutoTokenizer.from_pretrained('MLP-VLM/mllama_3b_stage2_base_10epoch')
sample_set = {'predictions':[example['predict'].strip() for example in examples],'references':[example['label'].strip() for example in examples]}
rouge_score = rouge.compute(predictions=sample_set['predictions'], references=sample_set['references'],tokenizer=lambda x: tokenizer.tokenize(x))

with open('/aihub-docviz/results/evaluation_result.json','w') as f:
    json.dump(rouge_score,indent=4,ensure_ascii=False)