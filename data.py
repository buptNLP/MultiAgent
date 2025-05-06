"""
GMS8k
Custom dataset for Llama3.
"""

from typing import Any
import jsonlines
from torch.utils.data import Dataset
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import numpy as np

    
class MathDataset(Dataset):
    
    def __init__(self, data_path, data_num=None):
        self.data = []
        
        cnt = 0
        with jsonlines.open(data_path, "r") as f:
            for line in f:
                self.data.append(line)
                cnt += 1
                if data_num is not None and cnt > data_num:
                    break
                
        for i in range(len(self.data)):
            self.data[i] = self.preprocess(self.data[i])

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def preprocess(self, sample):
        """
        Llama3 format.
        """
        
        question = sample["question"]
        answer = sample["answer"]
        
        
        instruction = f"{question} Please answer step by step and give the final answer(only the value of the result) after '#### '"
        response = f"{answer}"
        
        ret = {
            "question": instruction,
            "answer": response,
        }
        
        return ret
