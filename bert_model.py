from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import pkg_resources
import time
import scipy.stats as stats
import gc
import re 
import operator
import sys
import os 
from sklearn import metrics
from sklearn import model_selection
import torch 
import torch.nn as nn 
import torch.utils.data
import torch.nn.functional as F 
from nltk.stem import PorterStemmer
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm
import pandas as pd 
import numpy as np 
import shutil


device = torch.device('cuda')

MAX_SEQUENCE_LENGTH = 300
SEED = 1234
EPOCHS = 10
data_dir = './data/text_data.csv'
working_dir = './'
valid_size = 100000
num_to_load = 100000
target_column = 'target'

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam

BERT_MODEL_PATH = './bert_model/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + 'bert_model.ckpt',
BERT_MODEL_PATH + 'bert_config.json',
working_dir + 'pytorch_model.bin')

shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', working_dir + 'bert_config.json')

#bert config file
from pytorch_pretrained_bert import BertConfig

bert_config = BertConfig('../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'+'bert_config.json')

#convert text to bert format
def convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + [0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=True)
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

#change column name
train_df['text'] = train_df['text'].astype(str)

sequences = convert_lines(train_df['text'], MAX_SEQUENCE_LENGTH, tokenizer)
#change target column name
target_column = ['target']

X = sequences[:num_to_load]
X_Val = sequences[num_to_load:]
Y = train_df[target_column].values[:num_to_load]
Y_Val = train_df[target_column].values[num_to_load:]

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.float))

output_model_file = 'bert_pytorch.bin'

lr = 2e-5
batch_size = 32
accumulation_step = 2
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

model = BertForSequenceClassification.from_pretrained("./", cache_dir=None, num_labels=len(target_column))
model.zero_grad()
model = model.to(device)
params = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
train = train_dataset

num_train_optimization_step = int(EPOCHS * len(train)/batch_size/accumulation_step)

optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.5, t_total=num_train_optimization_step)

model = model.train()

tq = tqdm(range(EPOCHS))
for epoch in tq:
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
    optimizer.zero_grad()   # Bug fix - thanks to @chinhuic
    for i,(x_batch, y_batch) in tk0:
#        optimizer.zero_grad()
        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
        print('Y_PRED', y_pred)
        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
        loss.backward()
        if (i+1) % accumulation_step == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() / len(train_loader)
        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)


torch.save(model.state_dict(), output_model_file)

model = BertForSequenceClassification(bert_config,num_labels=len(target_column))
model.load_state_dict(torch.load(output_model_file ))
model.to(device)
for param in model.parameters():
    param.requires_grad=False
model.eval()
valid_preds = np.zeros((len(X_Val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)

tk0 = tqdm(valid_loader)
for i,(x_batch,)  in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    valid_preds[i*32:(i+1)*32]=pred[:,0].detach().cpu().squeeze().numpy()
    msle = mean_squared_log_error(Y_Val, valid_preds)

print('MSLE', msle)