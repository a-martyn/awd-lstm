import time
import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim

import model.net as net
from model.data_loader import Dictionary, tokenise, batch
from train import train, evaluate
from utils import epoch_metrics, NT_ASGD

from pprint import pprint


# Choose device
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
print(f'Running on: {device}')

# Globals
# --------------------------------------------------
path = './data/penn/'

batch_size = 20
emsize = 400
nhid = 1150


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

epochs = 3
lr = 0.4
timesteps = 35
clip = 0.25
weight_decay = 1.2e-6
non_monotone = 5

model = net.AWD_LSTM(ntokens, emsize, nhid, device=device).to(device)
# TODO: Check loss matches paper
criterion = nn.CrossEntropyLoss()
params = list(model.parameters()) + list(criterion.parameters())
nt_asgd = NT_ASGD(lr, weight_decay, non_monotone)

# TRAIN MODEL
# --------------------------------------------------

# set validation looss arbitrarily high initially 
# as hack to avoid ASGD triggering
val_loss = 100000000000000000000 

for epoch in range(1, epochs+1):
    epoch_start_time = time.time()
    optimizer = nt_asgd.get_optimizer(val_loss, params)
    train(model, train_data, criterion, optimizer, ntokens, batch_size, lr, timesteps, clip, device)
    val_loss = evaluate(model, val_data, criterion, ntokens, batch_size, timesteps, device)
    print(epoch_metrics(epoch, epoch_start_time, val_loss))









