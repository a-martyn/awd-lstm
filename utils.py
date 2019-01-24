import numpy as np
import math
import time
import torch as th
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


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
        metrics['memalloc_Gb'] = th.cuda.memory_allocated(device=device) / 1e+9
        metrics['memcache_Gb'] = th.cuda.memory_cached(device=device) / 1e+9
        metrics['max_memalloc_Gb'] = th.cuda.max_memory_allocated(device=device) / 1e+9
        metrics['max_memcache_Gb'] = th.cuda.max_memory_cached(device=device) / 1e+9
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

def batch_metrics2(start_time, device):
    metrics = {'batch_time': time.time() - start_time}
    if device == th.device('cuda:0'):
        metrics['memalloc_Gb'] = th.cuda.memory_allocated(device=device) / 1e+9
        metrics['memcache_Gb'] = th.cuda.memory_cached(device=device) / 1e+9
        metrics['max_memalloc_Gb'] = th.cuda.max_memory_allocated(device=device) / 1e+9
        metrics['max_memcache_Gb'] = th.cuda.max_memory_cached(device=device) / 1e+9
    return metrics

class NT_ASGD():
    """Non-monotonically triggered averaged stochastic gradient descent"""
    def __init__(self, lr, weight_decay, n):
        self.lr = lr
        self.weight_decay = weight_decay
        self.n = n

        self.asgd_triggered = False 
        self.losses = []
    
    def get_optimizer(self, val_loss):
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
        
        return self.asgd_triggered

#         # Return correct optimizer
#         if self.asgd_triggered:
#             optimizer = optim.ASGD(model_params, self.lr, t0=0, lambd=0, 
#                                    weight_decay=self.weight_decay)
#         else:
#             optimizer = optim.SGD(model_params, self.lr, 
#                                   weight_decay=self.weight_decay)
#         return optimizer


def plot_memory_usage(results_csv_filepath:str, output_filepath='./results/memory_plot.png'):
    """Plot memory usage per epoch to help spot memory leaks"""
    df = pd.read_csv(results_csv_filepath)
    x = df.index.values
    y = [df['memalloc_Gb'], df['memcache_Gb']]
    plt.stackplot(x, y, labels=['memalloc_Gb', 'memcache_Gb'])
    #plt.legend(loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('Gb')
    plt.title('Updated at: ' + str(datetime.now()))
    plt.savefig(output_filepath)
    return

def plot_memory_usage2(results_csv_filepath:str):
    """Plot memory usage per epoch to help spot memory leaks"""
    df = pd.read_csv(results_csv_filepath)
    x = df.index.values
    y = [df['memalloc_Gb']]
    plt.stackplot(x, y, labels=['memcache_Gb', 'memalloc_Gb'])
    #plt.legend(loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('Gb')
    plt.title('Updated at: ' + str(datetime.now()))
    plt.show()
    return

