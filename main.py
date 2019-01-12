import time
import numpy as np

import torch as th
import torch.nn as nn

import model.net as net
from model.data_loader import Dictionary, tokenise, batch
from train import train
from evaluate import evaluate



# Globals
# --------------------------------------------------

cuda = False
device = th.device("cuda" if cuda else "cpu")
path = './data/penn/'

batch_size = 40
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

model = net.AWD_LSTM(ntokens, emsize, nhid).to(device)
# TODO: Check loss matches paper
criterion = nn.CrossEntropyLoss()

epochs = 3
lr = 0.4
bptt = 35
clip = 0.25

for epoch in range(1, epochs+1):
    epoch_start_time = time.time()
    train(model, train_data, criterion, ntokens, batch_size, lr, bptt, clip)
    val_loss = evaluate(model, val_data, criterion, ntokens, batch_size, bptt)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, np.exp(val_loss)))
    print('-' * 89)









