import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch as th
import torch.nn as nn

from model.data_loader import get_batches, get_batch
from utils import batch_metrics, plot_memory_usage

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def repackage_hidden(hiddens):
    h_detached = [h.detach() for h in hiddens[0]]
    c_detached = [c.detach() for c in hiddens[1]]
    return (h_detached, c_detached)


def train(model, data, criterion, optimizer, ntokens:int, batch_size:int, 
          lr:float, timesteps:int, clip, device, alpha, beta):
    
    hiddens = model.init_hiddens(batch_size) 
    # Get base learning rate for scaling
    lr_base = optimizer.param_groups[0]['lr'] 
    
    batches = get_batches(data, timesteps, vary_seq_len=True)
    for inputs, targets in tqdm(batches):
        seq_len = len(inputs)
        start_time = time.time()
        
        # learning rate scaling based on seq_length
        # "necessary as sampling arbitrary sequence lengths with a fixed
        # learning rate favours short sequences over longer ones"
        # I've adapted this from the authors version which can result in 
        # inordinately large or small learning rate if seq_len is biased
        # above or below timesteps for many iterations
        lr_scaled = lr_base * seq_len / timesteps
        optimizer.param_groups[0]['lr'] = lr_scaled
        model.train()

        # For each batch, detach hidden state from state created in previous
        # batches. Else, the model would attempt backpropagation through the 
        # entire dataset
        hiddens = repackage_hidden(hiddens)
        # Zero the gradients from previous iteration, ready for new values
        optimizer.zero_grad()
        # Forward pass
        output, hiddens = model(inputs, hiddens)
        
        # Calculate loss
        # with Activation Regularisation and
        # Temporal Activation Regularisation
        loss = criterion(output.view(-1, ntokens), targets.view(-1))
        ar = model.activation_reg(alpha)
        tar = model.temporal_activation_reg(beta)
        loss = loss + ar + tar

        # Backpropagate
        loss.backward()
        
        # Gradient clipping
        # Note: criterion parameters aren't being clipped here
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        del loss
#         metrics = batch_metrics(start_time, device)
#         results_df = results_df.append(metrics, ignore_index=True)
#         results_df.to_csv('./results/batch_metrics.csv') 
        
    del hiddens
    return


def evaluate(model, data, criterion, ntokens, batch_size, timesteps, device):
    model.eval()
    total_loss = 0
    hiddens = model.init_hiddens(batch_size)
    with th.no_grad():
        for i in range(0, data.size(0) - 1, timesteps):
            inputs, targets, _ = get_batch(data, i, timesteps)
            output, hiddens = model(inputs, hiddens)
            total_loss += len(inputs) * criterion(model.decoder.weight, 
                                                  model.decoder.bias, output, 
                                                  targets).data
            hiddens = repackage_hidden(hiddens)
    return total_loss.item() / len(data)


