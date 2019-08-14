import torch 
import pandas as pd 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from nltk.stem import PorterStemmer
from tqdm import tqdm 

device = torch.device('cuda')

''' Многоклассовая классификация отрасли по новости, потом соотношения корелляции по заданной таблицы корреляции'''

'get company name ,  return corr coef beetwen '

'В первой итерации составим карту корелляция между всеми компаниями , для каждой выберет топ3 с которыми она коррелирована'

class Classifier():
    def __init__(self, company_name_list, corr_map):
        self.company_name_list = company_name_list
        self.corr_map = corr_map

    def is_about(self, text):
        '''
        return company_name that is text about
        '''
         #update to find in headline first, then in  abstract
        for company_name in self.company_name_list:
            if text.find(company_name) != -1:
                return company_name

    def classify(self, text, company_name):
        '''
        return: corr coef beetween text and company
        corr_matrix: map of maps? 
        '''
        text_company_name = self.is_about(text)
        return self.corr_map[text_company_name][company_name]