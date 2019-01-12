import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T

"""
Adapted from: https://github.com/pytorch/benchmark/blob/master/rnns/benchmarks/lstm_variants/lstm.py
"""

class LSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' Cell
    http://www.bioinf.jku.at/publications/older/2604.pdf

    a.k.a the Vanilla LSTM Cell

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Note: Dropout not needed in this class. The reference paper doesn't 
    implement dropout within an individual cell, only on recurrent weights 
    and activations between cells.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        
        # Contains all weights for the 4 linear mappings of the input x
        # e.g. Wi, Wf, Wo, Wc
        self.i2h = nn.Linear(input_size, 4*hidden_size, bias=bias)
        # Contains all weights for the 4 linear mappings of the hidden state h
        # e.g. Ui, Uf, Uo, Uc
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        # unpack tuple (recurrent activations, recurrent cell state)
        h, c = hidden

        # Linear mappings : all four in one hit
        preact = self.i2h(x) + self.h2h(h)

        # Activations
        i = F.sigmoid(preact[:, :self.hidden_size])                      # input gate
        f = F.sigmoid(preact[:, self.hidden_size:2*self.hidden_size])    # forget gate
        g = T.tanh(preact[:, 3*self.hidden_size:])                       # cell gate
        o = F.sigmoid(preact[:, 2*self.hidden_size:3*self.hidden_size])  # ouput gate


        # Cell state computations: 
        # calculates new long term memory state based on proposed updates c_T
        # and input and forget gate states i_t, f_t
        c_t = th.mul(f, c) + th.mul(i, g)

        # Output
        h_t = th.mul(o, F.tanh(c_t))

        return h_t, (h_t, c_t)


# How to handle batch? Is whole batch passed throug in a pass
# or iterate through?

class AWD_LSTM(nn.Module):

    """
    Constructs a 3 layer awd-lstm as described by:
    https://arxiv.org/abs/1708.02182

    """

    def __init__(self, ntokens, embedding_size, hidden_size, bias=True, dropout=0.5):
        super(AWD_LSTM, self).__init__()
        self.embedding = nn.Embedding(ntokens, embedding_size)
        self.layer1 = LSTMCell(embedding_size, hidden_size, bias=bias)
        self.layer2 = LSTMCell(hidden_size, hidden_size, bias=bias)
        self.layer3 = LSTMCell(hidden_size, hidden_size, bias=bias)
        self.decoder = nn.Linear(hidden_size, ntokens)

        self.nlayers = 3
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.hidden_size),
                weight.new_zeros(self.nlayers, batch_size, self.hidden_size))

    def forward(self, x, hiddens):
        # Translate input tokens to embedding vectors
        x = self.embedding(x)


        # LSTM
        # --------------
        # At each timestep t, forward propagate down through layers
        # Then proceed to next timestep for all timesteps.
        # Note: it would also be valid to go the other direction first.
        # e.g. iterate through all timesteps in layer 1 before proceeding
        # to layer 2
        h, c = hiddens
        output = T()
        # Forward pass for each timestep t
        # Note we adopt pytorch assumption that first dimension of x contains
        # so we pass batches all the way through
        for t in range(x.size(0)): 
            # Propagate through layers for each timestep
            # Note: using 3 layers here as per paper
            # .clone() is needed to avoid break in computation graph see:
            # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836
            zt_l1, (h[0,:,:], c[0,:,:]) = self.layer1(x[t,:,:].clone(), (h[0,:,:].clone(), c[0,:,:].clone()))
            zt_l2, (h[1,:,:], c[1,:,:]) = self.layer2(zt_l1,    (h[1,:,:].clone(), c[1,:,:].clone()))
            zt_l3, (h[2,:,:], c[2,:,:]) = self.layer3(zt_l2,    (h[2,:,:].clone(), c[2,:,:].clone()))
            # Record output from final layer at this timestep
            output = th.cat((output, zt_l3.unsqueeze(0)))

        # Translate embedding vectors to tokens
        reshaped = output.view(output.size(0)*output.size(1), output.size(2))
        decoded = self.decoder(reshaped)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

        return decoded, (h, c)




















