from torch.utils.data import Dataset
import torch
from os.path import join
import re
from collections import Counter
import pandas as pd


class MultiEmoDataset(Dataset):
    def __init__(self, filename):

        values = map(lambda x: x[:-1], re.findall(r'[a-z]+\.', filename))
        order = ['category', 'data_type', 'split', 'language']
        self.model_id = {
            order[i]: v for i, v in enumerate(values)
        }        
        
        self.filepath = filename
        self.data = self.parser()
        
        if isinstance(self.data, pd.DataFrame):
            self.X = self.data["DOCUMENT"].tolist()
            self.Y = self.data["TRUE_SENTIMENT"].tolist()
        else:
            self.X = [x[0] for x in self.data]
            self.Y = [x[1] for x in self.data]
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        curr_label = self.Y[index]
        y = self._process_labels_to_onehot(curr_label)
        return x, y
    
    @property
    def label_dict(self):
        return {
            'plus_m': 0,
            'minus_m': 1,
            'zero': 2,
            'amb': 3
        }
        
    @property
    def label_mark(self):
        if self.model_id['data_type'] == 'text':
            return '__label__meta_'
        return '__label__z_'

    def parser(self):
        if ".csv" in self.filepath:
            return pd.read_csv(self.filepath)[["DOCUMENT", "TRUE_SENTIMENT"]].dropna().reset_index(drop=True)
        with open(self.filepath, 'r') as f:
            data = f.readlines()
            data = [[s.rstrip() for s in x.split(self.label_mark)]
                    for x in data]
        return data
    
    def _process_labels_to_onehot(self, label):
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