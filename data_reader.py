import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import DataLoader, Dataset

#class DataReader should return text + fin data, if there is text in this day, or only fin data, if no

class FinancialDataReader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, sampler=None, batch_sampler=None,
    num_workers=0, collate_fn=None):
        self.dataset_obj = dataset #inherit torch Dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def rolling_window(self, feature, index=None, window_size=3, min_periods=1, win_type=None, func=None):
        #index - index in iteration process
        #func must get ndarray and return a single value
        #min_period == 1 allow to safe first {window_size} values without substitution them by NaN


        if index is None:
            data = self.dataset_obj.dataset
        else:
           data = self.dataset.__getitem__(index)   
        
        data[feature+'rolling_window'] = data[feature].rolling(window_size, min_periods=min_periods, win_type=win_type).apply(lambda x: func(x)).values
        return data

    

    
    

    


class SequenceDataSet(Dataset):
    def __init__(self, data, num_day):
        self.dataset = data
        self.num_day = num_day
        
    def __getitem__(self, index):
        return self.dataset.loc[index:index+self.num_day]

    def __len__(self):
        return self.dataset.shape[0]