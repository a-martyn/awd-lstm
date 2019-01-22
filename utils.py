import numpy as np
import math
import time
import torch as th
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def epoch_metrics(epoch, start_time, train_loss, val_loss, device):
    metrics = {
        'epoch': epoch,
        'time': time.time() - start_time,
        'train_loss': train_loss,
        'val_loss'  : val_loss,
        'train_ppl' : math.exp(train_loss),
        'val_ppl'   : math.exp(val_loss),
        'val_bpc'   : val_loss / math.log(2),
        'train_bpc' : train_loss / math.log(2)
    }
    # Get cuda memor metrics if device is cuda
    if device == th.device('cuda:0'):
        metrics['memalloc'] = th.cuda.memory_allocated(device=device)
        metrics['memcache'] = th.cuda.memory_cached(device=device)
        metrics['max_memalloc'] = th.cuda.max_memory_allocated(device=device)
        metrics['max_memcache'] = th.cuda.max_memory_cached(device=device)
    return metrics

def stringify(dictionary:dict):
    strings = [f'{k}: {v:.2f}' for k, v in dictionary.items()]
    return '| '.join(strings)   



def batch_metrics(batch, data, timesteps, lr, elapsed, log_interval, cur_loss):
    metrics = [f'| {batch}/{len(data) // timesteps} batches ',
               f'| lr {lr:05.5f} ',
               f'| ms/batch {elapsed * 1000 / log_interval:5.2f} ',
               f'| loss {cur_loss:5.2f} ',
               f'| ppl {np.exp(cur_loss):8.2f}',
               f'| bpc {cur_loss / math.log(2):8.3f}']
    return ''.join(metrics)


class NT_ASGD():
    """Non-monotonically triggered averaged stochastic gradient descent"""
    def __init__(self, lr, weight_decay, n):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n = n

        self.asgd_triggered = False 
        self.losses = []
    
    def get_optimizer(self, val_loss, model_params):
        n = self.n    # the non-monotone interval
        self.losses.append(val_loss)

        # Don't consider trigger condition until n+1
        # losses have been recorded
        if len(self.losses) < n+1:
            trigger = False
        else:
            trigger = self.losses[-n-1] < min(self.losses[-n:])
        
        # Switch to ASGD if loss hasn't improved for n timesteps
        # This is a one-way switch
        if not self.asgd_triggered and trigger:
            print('Switching to ASGD')
            self.asgd_triggered = True

        # Return correct optimizer
        if self.asgd_triggered:
            optimizer = optim.ASGD(model_params, self.lr, t0=0, lambd=0, 
                                   weight_decay=self.weight_decay)
        else:
            optimizer = optim.SGD(model_params, self.lr, 
                                  weight_decay=self.weight_decay)
        return optimizer

