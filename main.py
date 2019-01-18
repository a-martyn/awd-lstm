import time
import numpy as np
import pandas as pd

import torch as th
import torch.nn as nn
import torch.optim as optim

import model.net as net
from model.data_loader import Dictionary, tokenise, batch
from train import train, evaluate
from utils import epoch_metrics, stringify, NT_ASGD

from pprint import pprint

# Choose device
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
print(f'Running on: {device}')

# Globals
# --------------------------------------------------
path = './data/penn/'
MODEL_PATH = 'pretrained/awd_lstm.pt'
RESULTS_PATH = 'results/awd_lstm_fixed_seq_len.csv'
batch_size = 20


# LOAD DATA
# --------------------------------------------------

dictionary = Dictionary()

# Tokenise data to replace characters with integer indexes
train_data, dictionary = tokenise(path+'train.txt', dictionary)
val_data, dictionary   = tokenise(path+'valid.txt', dictionary)
test_data, dictionary  = tokenise(path+'test.txt', dictionary)

# Batch data: reshapes vector as matrix where number of columns j 
# is the batch size.
train_data = batch(train_data, batch_size)
val_data = batch(val_data, batch_size)
test_data  = batch(test_data, batch_size)

# Total number of tokens in corpus
ntokens = len(dictionary)

# Make sure all data is on GPU if available
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# INIT MODEL
# --------------------------------------------------

epochs = 500
lr = 0.4
timesteps = 35
emsize = 400
nhid = 1550
clip = 0.25
weight_decay = 1.2e-6
non_monotone = 5
dropout = 0.5   # authours use --dropouti 0.4 --dropouth 0.25
alpha = 2       # alpha L2 regularization on RNN activation (zero means no regularisation)
beta = 1        # beta slowness regularization applied on RNN activiation (zero means no regularisation)

model = net.AWD_LSTM(ntokens, emsize, nhid, dropout=dropout, device=device).to(device)
# TODO: Check loss matches paper
criterion = nn.CrossEntropyLoss()
params = model.parameters()
nt_asgd = NT_ASGD(lr, weight_decay, non_monotone)

# TRAIN MODEL
# --------------------------------------------------

# set validation looss arbitrarily high initially 
# as hack to avoid ASGD triggering
best_loss = 100000000000000000000
val_loss = 100000000000000000000 

cols = list(epoch_metrics(0, 0, 0, 0, device).keys())
results_df = pd.DataFrame(columns=cols).set_index('epoch')

for epoch in range(1, epochs+1):
    start_time = time.time()
    optimizer = nt_asgd.get_optimizer(val_loss, params)
    model_params = train(model, train_data, criterion, optimizer, ntokens, 
                         batch_size, lr, timesteps, clip, device, alpha, beta)
    params = list(model_params)
    
    # Record evaluation metrics
    # To save time just evaluate train_loss on first 3688 observations
    # this might be improved by random sampling i guess
    train_loss = evaluate(model, train_data[:3688], criterion, ntokens, batch_size, timesteps, device)
    val_loss   = evaluate(model, val_data, criterion, ntokens, batch_size, timesteps, device)
    metrics    = epoch_metrics(epoch, start_time, train_loss, val_loss, device)
    results_df.loc[epoch] = list(metrics.values())[1:]
    results_df.to_csv(RESULTS_PATH)
    print(stringify(metrics))

    # Save best model
    if val_loss < best_loss:
        print('Saving model')
        th.save(model.state_dict(), MODEL_PATH)
        best_loss = val_loss

        









