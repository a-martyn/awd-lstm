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


class WeightDropLSTM(nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


##########################################################################
# LSTM Model
##########################################################################


class LSTMModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # 'DropConnect' on recurrent hidden weights, but no dropout on LSTM outputs
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




