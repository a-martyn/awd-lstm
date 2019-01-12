import torch
import torch.nn as nn


##########################################################################
# DropConnect (a.k.a. WeightDrop)
##########################################################################

"""
The authors describe application DropConnect (Wang et al. 2013) to
the hidden-to-hidden weight matrices within the LSTM.

The following code accesses these weights within a Pytorch RNN module,
and then applies dropout.

This code is adapted a contribution to PetrochukM/PyTorch-NLP by the
papers authors: 
https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html

I've re-worked _weight_drop() to fix an apparent bug and added tests for WeightDrop()
and WeightDropLSTM()

"""

def _weight_drop(module, weights, dropout):
    """Helper function for WeightDrop class"""

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        state = module.state_dict()
        for name_w in weights:
            state[name_w] = nn.functional.dropout(state[name_w], p=dropout, training=module.training)

        module.load_state_dict(state)

        return original_module_forward(*args)

    setattr(module, 'forward', forward)

class WeightDrop(nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        module (:class:`torch.nn.Module`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
          dropout too.
        dropout (float): The probability a weight will be dropped.

    Example:

        >>> from torchnlp.nn import WeightDrop
        >>> import torch
        >>>
        >>> gru = torch.nn.GRUCell(2, 2)
        >>> weights = ['weight_hh']
        >>> weight_drop_gru = WeightDrop(gru, weights, dropout=0.9)
        >>> 
        >>> print('\nINITIAL STATE: \n')
        >>> print(weight_drop_gru.state_dict())
        >>>
        >>> # Forward pass
        >>> input_ = torch.randn(3, 2)
        >>> h0 = torch.randn(3, 2)
        >>> h1 = weight_drop_gru(input_, h0)
        >>> new_state = weight_drop_gru.state_dict()
        >>>
        >>> print('\nNEW STATE: \n')
        >>> print(weight_drop_gru.state_dict())
    """

    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward


class WD_LSTMCell(nn.LSTMCell):
    """
    Wrapper around :class:`torch.nn.LSTMCell` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh']
        _weight_drop(self, weights, weight_dropout)



class WeightDropLSTM(nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


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
    def __init__(self, example_input, p=0.5):
        super().__init__()
        ones = example_input.new_ones(example_input.size(), requires_grad=False)
        self.mask = nn.functional.dropout(ones, p=p)

    def forward(self, input):
        if not self.training:
            return input
        return input * self.mask




##########################################################################
# LSTM Model
##########################################################################



class AWD_LSTM_Layer(nn.Module):
    """
    I'm just implementing this layer explicitly for my own understanding
    but alternatievly could use the nn.LSTM() class
    """
    def __init__(self, input_size, hidden_size, weight_dropout=0.5):
        super(AWD_LSTM_Layer, self).__init__()
        self.wd_lstm_cell = WD_LSTMCell(ninp, nhid, weight_dropout=weight_dropout)

    def forward(self, inputs_, (output_prev, cellstate_prev)):
        hidden  = (output_prev, cellstate_prev)
        outputs = []
        for x in torch.unbind(input_, dim=1):
            output, cell_state = wd_lstm_cell(x, hidden)
            outputs.append(hidden[0].clone()) # clone to maintain computation graph

        return 










class AWD_LSTM(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(AWD_LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # "DropConnect" on recurrent hidden weights, but no dropout on LSTM outputs
        self.rnn = WeightDropLSTM(ninp, nhid, nlayers, dropout=0, weight_dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tie_weights flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        #output = self.drop(output)
        reshaped = output.view(output.size(0)*output.size(1), output.size(2))
        decoded = self.decoder(reshaped)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))




