import numpy as np 
import torch as th
import torch.nn as nn


def evaluate(model, data, criterion, ntokens, batch_size, bptt):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, bptt):
            x, y = get_batch(data, i)
            output, hidden = model(x, y)
            output_flat = output.view(-1, ntokens)
            total_loss += len(x) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data) - 1)