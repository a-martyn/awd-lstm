import os
from io import open
import torch


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
        tokens = torch.LongTensor(tokens)

    return tokens, dictionary


def batch(data, batch_size):
    """
    Reshape vector as matrix where number of columns j = batch size
    drops any remainder observations that don't fit perfectly.

    Expects data as PyTorch tensor, returns the same but reshaped.
    """
    # Calculate number of observations that remain after dividing dataset
    # into batches of size batch_size
    n = data.size(0)
    remainder = n % batch_size
    # Drop remainder observations 
    data = data[:-remainder]
    # Reshape vector into matrix where j=batch_size
    data = data.view(-1, batch_size).contiguous()
    return data

