import time
import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim

import model.net as net
from model.data_loader import Dictionary, tokenise, batch
from train import train
from evaluate import evaluate

from pprint import pprint


# Globals
# --------------------------------------------------

cuda = False
device = th.device("cuda" if cuda else "cpu")
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

# TRAIN A MODEL
# --------------------------------------------------

epochs = 3
lr = 0.4
timesteps = 35
clip = 0.25
weight_decay = 1.2e-6


model = net.AWD_LSTM(ntokens, emsize, nhid).to(device)
# TODO: Check loss matches paper
criterion = nn.CrossEntropyLoss()

params = list(model.parameters()) + list(criterion.parameters())
optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)

for epoch in range(1, epochs+1):
    epoch_start_time = time.time()
    train(model, train_data, criterion, optimizer, ntokens, batch_size, lr, timesteps, clip)
    val_loss = evaluate(model, val_data, criterion, ntokens, batch_size, timesteps)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, np.exp(val_loss)))
    print('-' * 89)









