import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T


"""
Adapted from: https://github.com/pytorch/benchmark/blob/master/rnns/benchmarks/lstm_variants/lstm.py
"""

class VariationalDropout(nn.Module):
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
    def __init__(self, p=0.5):
        super(VariationalDropout, self).__init__()
        self.p = p

    def forward(self, x, timestep):
        # Don't apply dropout if not training
        if not self.training:
            return x

        # Sample a new mask on first timestep only
        if timestep == 0:
            ones = x.new_ones(x.size(), requires_grad=False)
            self.mask = F.dropout(ones, p=self.p)
        return x * self.mask


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

        return h_t, (h_t, c_t)


# How to handle batch? Is whole batch passed throug in a pass
# or iterate through?

class AWD_LSTM(nn.Module):

    """
    Constructs a 3 layer awd-lstm as described by:
    https://arxiv.org/abs/1708.02182

    """

    def __init__(self, ntokens, embedding_size, hidden_size, bias=True, dropout=0.5, device='cpu'):
        super(AWD_LSTM, self).__init__()
        self.embedding = nn.Embedding(ntokens, embedding_size)
        self.layer0 = LSTMCell(embedding_size, hidden_size, bias=bias)
        self.layer1 = LSTMCell(hidden_size, hidden_size, bias=bias)
        self.layer2 = LSTMCell(hidden_size, embedding_size, bias=bias)
        self.decoder = nn.Linear(embedding_size, ntokens)

        self.nlayers = 3
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.device = device

        # Store activations for AR and TAR regularisation
        #TAR
#         self.output = None
#         self.output_nodrop = None

        # Dropout
        # TODO: expose dropout settings to api
        # QUESTION: How did authors arrive at these dropout parameters?
        self.dropout_wts = 0.5
        self.dropout_emb = 0.1
        self.dropout_inp = 0.4
        self.dropout_hid = 0.25
        self.varidrop_inp = VariationalDropout(p=self.dropout_inp)
        self.varidrop_hid = VariationalDropout(p=self.dropout_hid)
        self.varidrop_out = VariationalDropout(p=self.dropout_hid)

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
        h = [weight.new_zeros(batch_size, self.hidden_size),     # layer0
             weight.new_zeros(batch_size, self.hidden_size),     # layer1
             weight.new_zeros(batch_size, self.embedding_size)]  # layer2
        # cells state
        c = [weight.new_zeros(batch_size, self.hidden_size),     # layer0 
             weight.new_zeros(batch_size, self.hidden_size),     # layer1
             weight.new_zeros(batch_size, self.embedding_size)]  # layer2
        return (h, c)

    def weight_dropout(self, p=0.5):
        """
        Applies recurrent regularization through a DropConnect mask on the
        hidden-to-hidden recurrent weights as described here:
        https://cs.nyu.edu/~wanli/dropc/dropc.pdf
        """
        sd = self.state_dict()
        for l in range(self.nlayers):
            k = f'layer{l}.h2h.weight'
            sd[k] = F.dropout(sd[k], p=p, training=self.training)  
        self.load_state_dict(sd)
        return

    def embedding_dropout(self, embed, words, p=0.1):
        """
        Randomly drops whole word embeddings. As descride in section 
        4.3 Embedding Dropout of original paper.
        Largely adapted from: 
        https://github.com/salesforce/awd-lstm-lm/blob/master/embed_regularize.py
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


    def activation_reg(self, alpha):
        """
        Calulates a regularisation factor that increases with magnitude of
        activations from the final recurrent layer.
        See section 4.6 of paper: https://arxiv.org/abs/1708.02182
        
        Returns an integer that can be added to loss after each forward pass.
        """
        # The authors report that they apply the L_2 norm denoted ||.||_2
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


    def forward(self, x:th.LongTensor, hiddens):
        # Encode input
        # ----------------------------------------------------------------
        # Translate input tokens to embedding vectors
        # with dropout
        x = self.embedding_dropout(self.embedding, x, p=self.dropout_emb)

        # LSTM
        # ----------------------------------------------------------------
        # Apply DropConnect to hidden-to-hidden weights 
        # once for each forward pass
        self.weight_dropout(p=self.dropout_wts)


        # At each timestep t, forward propagate down through layers
        # Then proceed to next timestep for all timesteps.
        # Note: it would also be valid to go the other direction first.
        # e.g. iterate through all timesteps in first layer before proceeding
        # to next layer
        # Note: we adopt pytorch assumption that first dimension of x contains
        # so we pass batches all the way through

        h, c = hiddens
        output = T().to(self.device)
        #TAR
        #output_nodrop = T().to(self.device)
        for t in range(x.size(0)): 
            # Propagate through layers for each timestep
            # Note: using 3 layers here as per paper
            # .clone() is needed to avoid break in computation graph see:
            # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836
            inp = x[t,:,:]
            inp_d = self.varidrop_inp(inp, t)
            z0, (h0, c0) = self.layer0(inp_d, (h[0], c[0]))
            z0_d = self.varidrop_hid(z0, t)
            z1, (h1, c1) = self.layer1(z0_d, (h[1], c[1]))
            z1_d = self.varidrop_hid(z1, t)
            z2, (h2, c2) = self.layer2(z1_d, (h[2], c[2]))
            # Note: Can't use the same variational dropout mask here because
            # the final layer outputs a different sized matrix.
            z2_d = self.varidrop_out(z2, t)
            h = [h0, h1, h2]
            c = [c0, c1, c2]
            output = th.cat((output, z2_d.unsqueeze(0)))
            #TAR
            #output_nodrop = th.cat((output_nodrop, z2.unsqueeze(0)))
    
        # Store outputs for AR and TAR regularisation
        # Detach because we don't want subequent calcs to affect
        # backpropagation
        #TAR
        #self.output = output.detach()
        #self.output_nodrop = output_nodrop.detach()
        
        # Decode output
        # ----------------------------------------------------------------
        # Translate embedding vectors to tokens
        reshaped = output.view(output.size(0)*output.size(1), output.size(2))
        decoded = self.decoder(reshaped)
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

        return decoded, (h, c)




















