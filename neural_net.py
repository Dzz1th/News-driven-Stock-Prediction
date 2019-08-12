import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.model_selection import TimeSeriesSplit
from data_reader import DataReader

'''There will be a fine-tuned on Twitter Bert for sentiment analysis, fully-connected NN , and classifier upon the BERT,
that will show correlation beetwen news and this stock(need stock name), if it more than any threshold, than we use
sentiment coef * correlation factor and use it in out fully connected NN'''

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
        text = get_text(x)
        company_name = get_company_name(x)
        fin_data = get_fin_data(x)
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


if __name__ == '__main__':
    config = {
        'EPOCHS':100,
        'num_financial_features':6,
        'nn_config':{
            'BertModel': bertModel,
            'classifier': classifier, 
            'num_features': 6
        },
        'log_interval': 10
    }
    optimizer = torch.optim.Adam(NeuralNet.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    Net = NeuralNet(config['nn_config'])
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
                    epoch, batch_idx* len(x_batch), len(DataReader.dataset),
                           100. * batch_idx/ len(DataReader), loss.data[0]))
                    
    test_loss=0
    correct=0
    for batch_idx, (x_batch, y_batch) in enumerate(TestDataReader):
        x_batch = x_batch.view(-1, config['num_features'])
        out = Net(x_batch)
        text_loss += criterion(out, y_batch).data[0]
    
    test_loss /= TestDataReader.dataset
    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss)


