import torch

class LlavaNextDataCollator:
    def __init__(self, processor, ignore_index=-100):
        self.processor = processor
        self.ignore_index = ignore_index
        self.assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids('assistant')
    
        
    def __call__(self,examples):
        texts = [self.processor.tokenizer.apply_chat_template(example['messages'],tokenize=False) for example in examples]
        images = [example['images'] for example in examples]
        
        batch = self.processor(
            images=images,
            text=texts,
            padding=True,
            return_tensors='pt',   
        )
         
        labels = batch['input_ids'].clone()
        batch_indices,token_indices = torch.where(labels==self.assistant_token_id)
        
        for batch_idx,token_idx in zip(batch_indices,token_indices):
            labels[batch_idx][:token_idx+3] = self.ignore_index
            
        batch['labels']=labels

        return batch
        
        
        
        
            
            
        
        