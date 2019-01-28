
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
import warnings

from pprint import pprint


class VariationalDropout():
    """ An adaption of torch.nn.functional.dropout that applies 
    the same dropout mask each time it is called.

    Samples a binary dropout mask only once upon instantiatin and then 
    allows that same dropout mask to be used repeatedly. When minibatches
    are received as input, then a different mask is used for each minibatch.

    Described in section 4.2 of the AWD-LSTM reference paper where they cite:
    A Theoretically Grounded Application of Dropout in Recurrent Neural Networks 
    (Gal & Ghahramani, 2016, https://arxiv.org/abs/1512.05287)

    TODO Note: The AWD-LSTM authors' implementation is not as described in paper.
    There code appears to sample a new mask on each call.
    https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    """
    def __init__(self):
        pass

    def apply(self, x, timestep, training, p=0.5):
        # Don't apply dropout if not training
        if not training:
            return x

        # Sample a new mask on first timestep only
        if timestep == 0:
            ones = x.new_ones(x.size(), requires_grad=False)
            self.mask = F.dropout(ones, p=p)
        return x * self.mask


class WeightDropout(nn.Module):
    """
    A module that wraps an LSTM cell in which some weights will be replaced by 0 during training.
    Adapted from: https://github.com/fastai/fastai/blob/master/fastai/text/models.py
    
    Initially I implemented this by getting the models state_dict attribute, modifying it to drop
    weights, and then loading the modified version with load_state_dict. I had to abandon this 
    approach after identifying it as the source of a slow memory leak.
    """

    def __init__(self, module:nn.Module, weight_p:float):
        super().__init__()
        self.module,self.weight_p = module, weight_p
            
        #Makes a copy of the weights of the selected layers.
        w = getattr(self.module.h2h, 'weight')
        self.register_parameter('weight_raw', nn.Parameter(w.data))
        self.module.h2h._parameters['weight'] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        raw_w = getattr(self, 'weight_raw')
        self.module.h2h._parameters['weight'] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        raw_w = getattr(self, 'weight_raw')
        self.module.h2h._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()


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

    def __init__(self, input_size, output_size, bias=True):
        super(LSTMCell, self).__init__()
        
        # Contains all weights for the 4 linear mappings of the input x
        # e.g. Wi, Wf, Wo, Wc
        self.i2h = nn.Linear(input_size, 4*output_size, bias=bias)
        # Contains all weights for the 4 linear mappings of the hidden state h
        # e.g. Ui, Uf, Uo, Uc
        self.h2h = nn.Linear(output_size, 4*output_size, bias=bias)
        self.output_size = output_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.output_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        # unpack tuple (recurrent activations, recurrent cell state)
        h, c = hidden

        # Linear mappings : all four in one vectorised computation
        preact = self.i2h(x) + self.h2h(h)

        # Activations
        i = th.sigmoid(preact[:, :self.output_size])                      # input gate
        f = th.sigmoid(preact[:, self.output_size:2*self.output_size])    # forget gate
        g = th.tanh(preact[:, 3*self.output_size:])                       # cell gate
        o = th.sigmoid(preact[:, 2*self.output_size:3*self.output_size])  # ouput gate


        # Cell state computations: 
        # calculates new long term memory state based on proposed updates c_T
        # and input and forget gate states i_t, f_t
        c_t = th.mul(f, c) + th.mul(i, g)

        # Output
        h_t = th.mul(o, th.tanh(c_t))

        return h_t, c_t


class AWD_LSTM(nn.Module):

    """
    Constructs a 3 layer awd-lstm as described by:
    https://arxiv.org/abs/1708.02182
    """

    def __init__(self, ntokens, embedding_size, hidden_size, bias=True, device='cpu',
                 dropout_wts=0.5, dropout_emb=0.1, dropout_inp=0.4, dropout_hid=0.25):
        super(AWD_LSTM, self).__init__()
        
        self.dropout_emb = dropout_emb
        self.dropout_inp = dropout_inp
        self.dropout_hid = dropout_hid
        
        self.embedding = nn.Embedding(ntokens, embedding_size)
        self.layer0 = LSTMCell(embedding_size, hidden_size, bias=bias)
        self.layer0 = WeightDropout(self.layer0, dropout_wts)
        self.layer1 = LSTMCell(hidden_size, hidden_size, bias=bias)
        self.layer1 = WeightDropout(self.layer1, dropout_wts)
        self.layer2 = LSTMCell(hidden_size, embedding_size, bias=bias)
        self.layer2 = WeightDropout(self.layer2, dropout_wts)
        self.decoder = nn.Linear(embedding_size, ntokens)
        
        self.varidrop_inp = VariationalDropout()
        self.varidrop_hid = VariationalDropout()
        self.varidrop_out = VariationalDropout()

        self.nlayers = 3
        self.hidden_size = hidden_size
        self.device = device
        self.embedding_size = embedding_size
        
        self.output = None
        self.output_nodrop = None
        
        # Weight tying
        # https://arxiv.org/abs/1608.05859
        # https://arxiv.org/abs/1611.01462
        #self.decoder.weight = self.embedding.weight


    def init_hiddens(self, batch_size):
        """
        Create initial tensors as input to timestep 0
        for each of the layers.
        """
        weight = next(self.parameters())
        # hidden activations
        h = [weight.new_zeros(batch_size, self.hidden_size).to(self.device),     # layer0
             weight.new_zeros(batch_size, self.hidden_size).to(self.device),     # layer1
             weight.new_zeros(batch_size, self.embedding_size).to(self.device)]  # layer2
        # cells state
        c = [weight.new_zeros(batch_size, self.hidden_size).to(self.device),     # layer0 
             weight.new_zeros(batch_size, self.hidden_size).to(self.device),     # layer1
             weight.new_zeros(batch_size, self.embedding_size).to(self.device)]  # layer2
        return (h, c)
    
    def embedding_dropout(self, embed, words, p=0.1):
        """
        TODO: re-write and add test
        """
        if not self.training:
            masked_embed_weight = embed.weight
        elif not p:
            masked_embed_weight = embed.weight
        else:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - p).expand_as(embed.weight) / (1 - p)
            masked_embed_weight = mask * embed.weight
    
        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1
    
        X = F.embedding(words, masked_embed_weight,
                        padding_idx, embed.max_norm, embed.norm_type,
                        embed.scale_grad_by_freq, embed.sparse)
        return X

    def forward(self, x, hiddens):
        # Translate input tokens to embedding vectors
        # with dropout
        x = self.embedding_dropout(self.embedding, x, p=self.dropout_emb)
        
        h, c = hiddens
        output = T().to(self.device)
        output_nodrop = T().to(self.device)
        for t in range(x.size(0)):          
            # Propagate through layers for each timestep
            # Note: using 3 layers here as per paper
            inp    = x[t,:,:]
            inp_d  = self.varidrop_inp.apply(inp, t, self.training, p=self.dropout_inp)
            h0, c0 = self.layer0(inp_d, (h[0], c[0]))
            z0     = self.varidrop_hid.apply(h0, t, self.training, p=self.dropout_hid) 
            h1, c1 = self.layer1(z0, (h[1], c[1]))
            z1     = self.varidrop_hid.apply(h1, t, self.training, p=self.dropout_hid) 
            h2, c2 = self.layer2(z1, (h[2], c[2]))
            # Note: Can't use the same variational dropout mask here because
            # the final layer outputs a different sized matrix.
            z2 = self.varidrop_out.apply(h2, t, self.training, p=self.dropout_hid)
            h = [h0, h1, h2]
            c = [c0, c1, c2]
            output = th.cat((output, z2.unsqueeze(0)))
            #TAR
            output_nodrop = th.cat((output_nodrop, h2.unsqueeze(0)))
            
        # Store outputs for AR and TAR regularisation
        # Detach because we don't want subequent calcs to affect
        # backpropagation
        self.output = output.detach()
        self.output_nodrop = output_nodrop.detach()
        
        # Translate embedding vectors to tokens
        reshaped = output.view(output.size(0)*output.size(1), output.size(2))
        decoded = self.decoder(reshaped)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

        return decoded, (h, c)


    def activation_reg(self, alpha):
        """
        Calulates a regularisation factor that increases with magnitude of
        activations from the final recurrent layer.
        See section 4.6 of paper: https://arxiv.org/abs/1708.02182
        
        Returns an integer that can be added to loss after each forward pass.
        """
        # The authors report that they apply the L_2 norm denoted ||.||_2
        # but they actually implement code equivalent to the following
        # which is missig a square root operation:
        # ar = sum(alpha * a.pow(2).mean() for a in self.output)
        # return ar
        # I assume that the mean is intended to be across timesteps
        # and also across items in this batch
        # So we want the mean L2 norm across all timesteps, across all
        # items in this mini-batch. Verbosely this can be written as:
        # ar = [th.sqrt(th.sum(a.pow(2), dim=1)).mean() for a in self.output]
        # return alpha * (sum(ar)/len(ar)))
        # Using pytorch's built in torch.norm 
        masked_ht = self.output
        L2_t = [th.norm(a, dim=1, p='fro').mean() for a in masked_ht]
        return alpha * th.mean(T(L2_t)).item()


    def temporal_activation_reg(self, beta):
        # Not sure if this would be better returning average (as current) 
        # or max?
        ht = self.output_nodrop
        L2_norms_t = []
        for timestep in range(ht.size(0)-1):
            diff = ht[timestep] - ht[timestep+1]
            L2_norm_per_batch = th.norm(diff, dim=1, p='fro')
            L2_norm_mean = th.mean(L2_norm_per_batch).item()
            L2_norms_t.append(L2_norm_mean)
        
        return beta * th.mean(T(L2_norms_t))











