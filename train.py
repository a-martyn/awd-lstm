import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch as th
import torch.nn as nn
import gc

from model.data_loader import get_batch
from utils import batch_metrics2, plot_memory_usage

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history"""
    if isinstance(h, th.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


    
def train(model, data, criterion, optimizer, ntokens:int, batch_size:int, 
          lr:float, timesteps:int, clip, device, alpha, beta, results_df):
    hiddens = model.init_hiddens(batch_size)
    model.train()
    #hidden = (h.to(device) for h in hidden)

    # TODO: Should this now be a while loop to account for variable
    # sequence length?
    for batch, i in tqdm(enumerate(range(0, data.size(0)-1, timesteps))):
        # Get batches for this epoch with variable sequence length
        inputs, targets, seq_len = get_batch(data, i, timesteps, jitter=False)

        # learning rate scaling based on seq_length
        # "necessary as sampling arbitrary sequence lengths with a fixed
        # learning rate favours short sequences over longer ones"
#         lr2 = optimizer.param_groups[0]['lr'] 
#         optimizer.param_groups[0]['lr'] = lr2 * seq_len / timesteps

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
        # ar  = model.activation_reg(alpha)
        # tar = model.temporal_activation_reg(beta)
        # loss = loss + ar + tar

        # Backpropagate
        loss.backward()
        
        # Gradient clipping
        # Note: criterion parameters aren't being clipped here
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        metrics = batch_metrics2(device)
        results_df = results_df.append(metrics, ignore_index=True)
        results_df.to_csv('./results/batch_metrics.csv')
        #plot_memory_usage('./results/batch_metrics.csv', output_filepath='./results/batch_memory.png')
        
        elements = 0
        for obj in gc.get_objects():
            try:
                if th.is_tensor(obj) or (hasattr(obj, 'data') and th.is_tensor(obj.data)):
                    m = 1
                    for s in obj.size():
                        m = m*s
                    elements += m
                        
            except:
                pass
        print(elements)
        
        if batch > 2:
            break
        
        del loss
        
    del hiddens
    return results_df




def evaluate(model, data, criterion, ntokens, batch_size, timesteps, device):
    model.eval()
    total_loss = 0
    hiddens = model.init_hiddens(batch_size)
    #hidden = (h.to(device) for h in hidden)
    with th.no_grad():
        for i in range(0, data.size(0) - 1, timesteps):
            inputs, targets, _ = get_batch(data, i, timesteps)
            output, hiddens = model(inputs, hiddens)
            total_loss += len(inputs) * criterion(output.view(-1, ntokens), targets.view(-1)).item()
            hiddens = repackage_hidden(hiddens)
    return total_loss / (len(data) - 1)


