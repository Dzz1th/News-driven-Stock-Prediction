import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import DataLoader, Dataset

#class DataReader should return text + fin data, if there is text in this day, or only fin data, if no


class DataReader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, sampler=None, batch_sampler=None,
    num_workers=0, collate_fn=None):
        self.dataset_obj = dataset #inherit torch Dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    



    
class CompanyDataset(Dataset):
    def __init(self, company_news_data, company_financial_data, lag):
        self.company_news_data = company_news_data
        self.company_financial_data = company_financial_data
        self.lag = lag

    def __getitem__(self, index):
        return pd.concat([self.company_news_data[self.company_news_data['date']==index],
     self.company_financial_data[(self.company_financial_data['date']<(index+self.lag))&(self.company_financial_data['date']>=index)]],
      axis=1)
        
    def __len__(self):
        return self.company_financial_data.shape[0] + self.company_news_data.shape[0]


class DatasetFactory():
    def __init__(self, news_data, financial_data, company_id, company_name, lag):
        self.news_data = news_data
        self.financial_data = financial_data
        self.company_id = company_id
        self.company_name = company_name
        self.lag = lag
    
    def create_company_dataset(self):
        company_news_data = self.news_data[is_about(self.news_data['HEADLINE'], self.company_name)] #implement is_about func
        company_financial_data = self.financial_data[self.financial_data['ws_id'] == self.company_id]
        return CompanyDataset(company_news_data, company_financial_data, self.lag)

    def set_company_name(self, company_name):
        self.company_name = company_name

    def set_company_id(self, company_id):
        self.company_id = company_id

    def set_lag(self, lag):
        self.lag = lag