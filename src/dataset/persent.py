from torch.utils.data import Dataset
import torch
from os.path import join
import re
import pandas as pd
from collections import Counter


class PerSenTDataset(Dataset):
    def __init__(self, filename):      
        self.data = pd.read_csv(filename)[["DOCUMENT", "TRUE_SENTIMENT"]].dropna().reset_index(drop=True)
        
        self.X = self.data["DOCUMENT"]
        self.Y = self.data["TRUE_SENTIMENT"]
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        curr_label = self.Y[index]
        y = self._process_label_to_onehot(curr_label)
        return x, y
    
    @property
    def label_dict(self):
        return {
            'Positive': 0,
            'Negative': 1,
            'Neutral': 2,
        }
    
    def _process_label_to_onehot(self, label):
        onehot = torch.zeros(len(self.label_dict), dtype=torch.float32)
        onehot[self.label_dict[label]] = 1.
        return onehot
    
    def class_proportion(self):
        count = Counter(self.Y)
        total = len(self.Y)
        proportion = torch.zeros(len(self.label_dict))
        for k, v in count.items():
            proportion[self.label_dict[k]] = v/total
        return proportion