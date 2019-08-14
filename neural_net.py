import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.model_selection import TimeSeriesSplit
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from data_reader import DatasetFactory
from classifier import Classifier 

'''There will be a fine-tuned on Twitter Bert for sentiment analysis, fully-connected NN , and classifier upon the BERT,
that will show correlation beetwen news and this stock(need stock name), if it more than any threshold, than we use
sentiment coef * correlation factor and use it in out fully connected NN'''

company_names = {
    '1001': 'Microsoft'
}

class NeuralNet(nn.Module):

    def __init__(self, nn_config):
        #num_features is fin_features + 1(bert final output)
        super(NeuralNet, self).__init__()
        self.text_model = nn_config['BertModel']
        self.classifier = nn_config['classifier']
        self.num_features = nn_config['num_features']
        self.fc_layer = nn.Linear(self.num_features, self.num_features)
        self.bn_layer = nn.BatchNorm1d(self.num_features)
        self.last_layer = nn.Linear(self.num_features, 1)

    def forward(self, x):
        length = len(x)
        text = self.get_text(x)
        company_name = self.get_company_name(x)
        fin_data = self.get_fin_data(x)
        if text:
            text_model_out = self.text_model(text)
            corr = classifier(text, company_name)
            result = text_model_out * corr
            result_vector = fin_data.append(result)
        else:
            result_vector = fin_data.append(0)

        if len(result_vector) == self.num_features:
            result_vector= F.relu(self.fc_layer(result_vector))
            result_vector= self.bn_layer(result_vector)
            result_vector= F.relu(self.fc_layer(result_vector))
            result_vector= self.bn_layer(result_vector)
            result_vector= self.last_layer(result_vector)
            return result_vector
        else:
            raise AttributeError('input len should be equal to num_features+num_financial_features, your len is {length} ')
    
    def get_text(self, x):
        '''
        params: x - pandas dataFrame with news data and fin data
        return: data frame with 'HEADLINE', 'ABSTRACT'
        '''
        return x[['HEADLINE', 'ABSTRACT']]

    def get_company_name(self, x):
        '''
        return company name using it ws_id in fin data
        params: x- pandas data Frame with news data and fin data
        return: return lowercase company name
        '''
        return company_names[x['ws_id']].lower()

    def get_fin_data(self, x):
        return x.drop(['HEADLINE', 'ABSTRACT', 'ws_id'], axis=1)

if __name__ == '__main__':
    config = {
        'EPOCHS':100,
        'news_data_path': './data/news.csv',
        'financial_data_path':'./data/fin.csv',
        'num_financial_features':6,
        'nn_config':{
            'BertModel': bertModel,
            'classifier': classifier, 
            'num_features': 6
        },
        'log_interval': 10,
        'company_name': 'Apple',
        'company_id':'1000',
        'batch_size':32
    }
    Net = NeuralNet(config['nn_config'])
    optimizer = torch.optim.Adam(NeuralNet.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    TrainDataReader = DataLoader(DatasetFactory(config['news_data_path'], config['financial_data'], config['company_id'], config['company_name'], config['log_interval']).create_company_dataset(),
     batch_size=config['batch_size'])
    for epoch in range(config['EPOCHS']):
        for batch_idx, (x_batch, y_batch) in enumerate(TrainDataReader):
            #if has no text in data, then 1-st feature == 0
            #resize data from (batch_size, 1, num_financial_features) to (batch_size, num_financial_features)
            x_batch = x_batch.view(-1, config['num_features']) 
            optimizer.zero_grad()
            out = Net(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            if batch_idx % config['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx* len(x_batch), len(TrainDataReader.dataset),
                           100. * batch_idx/ len(TrainDataReader), loss.data[0]))
                    
    test_loss=0
    correct=0
    for batch_idx, (x_batch, y_batch) in enumerate(TestDataReader):
        x_batch = x_batch.view(-1, config['num_features'])
        out = Net(x_batch)
        text_loss += criterion(out, y_batch).data[0]
    
    test_loss /= TestDataReader.dataset
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))



    #Перенести логику выделение текстовой и финансовой составляющей в DataReader