import time
import numpy as np
from tqdm import tqdm

import torch as th
import torch.nn as nn

from model.data_loader import get_batch
from utils import batch_metrics


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history"""
    if isinstance(h, th.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(model, data, criterion, optimizer, ntokens:int, batch_size:int, lr:float, timesteps:int, clip, device):
    model.train()
    hiddens = model.init_hiddens(batch_size)
    #hidden = (h.to(device) for h in hidden)
    for batch, i in tqdm(enumerate(range(0, data.size(0)-1, timesteps))):
        inputs, targets = get_batch(data, i, timesteps)
        # For each batch, detach hidden state from state created in previous
        # batches. Else, the model would attempt backpropagation through the 
        # entire dataset
        hiddens = repackage_hidden(hiddens)
        # Zero the gradients from previous iteration, ready for new values
        optimizer.zero_grad()
        # Forward pass
        output, hiddens = model(inputs, hiddens)
        # Calculate loss
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
        # Backpropagate
        loss.backward()
        
        # Gradient clipping
        # Note: criterion parameters aren't being clipped here
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return model.parameters()




def evaluate(model, data, criterion, ntokens, batch_size, timesteps, device):
    model.eval()
    total_loss = 0
    hiddens = model.init_hiddens(batch_size)
    #hidden = (h.to(device) for h in hidden)
    with th.no_grad():
        for i in range(0, data.size(0) - 1, timesteps):
            inputs, targets = get_batch(data, i, timesteps)
            output, hiddens = model(inputs, hiddens)
            total_loss += len(inputs) * criterion(output.view(-1, ntokens), targets.view(-1)).item()
            hiddens = repackage_hidden(hiddens)
    return total_loss / (len(data) - 1)


