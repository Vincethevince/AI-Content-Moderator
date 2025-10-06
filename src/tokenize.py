from transformers import BertTokenizer
from multiprocessing import Pool
import numpy as np

class MyTokenizer:
    def __init__(self, pretrained_name='bert-base-uncased', max_length=128, num_workers=1):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        self.max_length = max_length
        self.num_workers = num_workers

    def _tokenize_single(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='np')

    def tokenize_batch(self, texts):
        with Pool(self.num_workers) as p:
            results = p.map(self._tokenize_single, texts)
        input_ids = np.vstack([r['input_ids'] for r in results])
        attention_mask = np.vstack([r['attention_mask'] for r in results])
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask}