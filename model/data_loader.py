import os
from io import open
import numpy as np
import torch as th
import torch.tensor as T
from scipy.stats import truncnorm


class Dictionary(object):

    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = []

    def __len__(self):
        return len(self.idx_to_word)


def add_words(words:list, dictionary):
    """Add words to dictionary if not already listed"""
    for word in words:
        if word not in dictionary.word_to_idx:
            dictionary.idx_to_word += [word]
            dictionary.word_to_idx[word] = len(dictionary) - 1
    return dictionary


def tokenise(path, dictionary):
    """
    Tokenise .txt file a path provided
    - if no dictionary is provided then a new one is created
    - else words in corpus that aren't in dictionary are added to dictionary.
    - End of sentences are tagged <eos>

    Returns tokenised corpus as a PyTorch tensor along with
    the updated dictionary.
    """

    # Update dictionary
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            words = line.split() + ['<eos>']
            dictionary = add_words(words, dictionary)
    
    # Tokenise file
    with open(path, 'r', encoding='utf8') as f:
        tokens = []
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                tokens += [dictionary.word_to_idx[word]]
        tokens = th.LongTensor(tokens)

    return tokens, dictionary


def batch(data, batch_size):
    """
    Reshape vector as matrix where number of columns j = batch size
    drops any remainder observations that don't fit perfectly.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘

    Expects data as PyTorch tensor, returns the same but reshaped. See tests.py

    Adapted from: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    """
    # Calculate number of observations that remain after dividing dataset
    # into batches of size batch_size
    n = data.size(0)
    remainder = n % batch_size
    # Drop remainder observations 
    data = data.narrow(0, 0, n-remainder)
    # Reshape vector into matrix where j=batch_size
    data = data.view(batch_size, -1).t().contiguous()
    return data


def get_batch(data, i, seq_len, jitter=False):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If data is equal to the example output of the batch function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘

    See test for details

    Adapted from: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    """
    if jitter:
        seq_len = seq_len if np.random.random() < 0.95 else seq_len/2
        # prevent excessively small or negative lengths
        seq_len = max(5, int(np.random.normal(seq_len, 5)))

    seq_len = min(seq_len, len(data)-1-i)
    x = data[i:i+seq_len]
    y = data[i+1:i+1+seq_len]
    return x, y, seq_len


def gen_sequence_lengths(data, base_seq_len):
    # sample all jittered sequence lengths
    # slice data by sequence lengths to produce batches
    total_data_len = data.size(0)
    seq_lens = []
    total_seq_len = 0
    while total_seq_len < total_data_len:
        # occasionally halve the base sequence length
        mu = base_seq_len if np.random.random() < 0.95 else base_seq_len/2
        # prevent excessively small or negative lengths
        seq_len = max(5, int(np.random.normal(mu, 5)))
        # add seqence lengths until total data length is exceeded
        seq_lens.append(seq_len)
        total_seq_len += seq_len
    # remove last item because it exceed total_data_len
    seq_lens = seq_lens[:-1]
    # add final sequence that fits perfectly if greater than 5
    remainder = total_data_len - sum(seq_lens)
    if remainder > 5:
        seq_lens.append(remainder)
    
    return seq_lens


def get_batches(data, base_seq_len):
    # Get seq lens
    seq_lens = gen_sequence_lengths(data, base_seq_len)
    # slice data into batches by seq_lens
    batches = []
    for i in range(len(seq_lens)):
        seq_start = sum(seq_lens[:i-1])
        seq_end = seq_start + seq_lens[i]
        batches.append(data[seq_start: seq_end])

    # order batches by descending sequence length
    # this step is important to avoid excessive memory caching by pytorch
    # if the first tensor created by model is the largest required
    # then a new cache isn't required because model will never receive a larger
    # input than its first. See discussion here:
    # https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/11
    # https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
    batches.sort(key=len, reverse=True)

    return batches



