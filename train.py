import time
import numpy as np

import torch as th
import torch.nn as nn

from model.data_loader import get_batch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history"""
    if isinstance(h, th.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(model, data, criterion, optimizer, ntokens:int, batch_size:int, lr:float, bptt:int, clip):
    log_interval = 1
    
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, data.size(0)-1, bptt)):
        inputs, targets = get_batch(data, i, bptt)
        # For each batch, detach hidden state from state created in previous
        # batches. Else, the model would attempt backpropagation through the 
        # entire dataset
        hidden = repackage_hidden(hidden)
        # Zero the gradients from previous iteration, ready for new values
        optimizer.zero_grad()
        # Forward pass
        output, hidden = model(inputs, hidden)
        # Calculate loss
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # TODO: Check clipping config
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)
            
        total_loss += loss.item()
        
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed  = time.time() - start_time
            print('| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                batch, len(data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, np.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    



