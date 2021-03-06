import unittest
import string
import torch as th
from torch import Tensor as T

# Local
import net


class Tests(unittest.TestCase):



    def test_activation_reg(self):
        # make deterministe
        th.manual_seed(0) 
        # some dummy params
        emb_size = 3
        alpha = 1
        model = net.AWD_LSTM(10, emb_size, 2)
        # manually set the output to some random values
        # in practice this is done automatically on each forward pass
        # expects model output of shape (timesteps, batch_size, embedding_size)
        # batch_size = 2
        # timesteps = 3
        model.output = T([[[3, 4, 5],
                           [3, 4, 5]],
                          [[3, 4, 5],
                           [3, 4, 5]],
                          [[3, 4, 5],
                           [3, 4, 5]]])
    
        expect = (T([[3, 4, 5]]) @ T([[3, 4, 5]]).t()).pow(0.5)
        # equivalently
        # expect = th.sum(T([3, 4, 5]).pow(2)).pow(0.5)
        ar = model.activation_reg(alpha)
        self.assertEqual(ar, expect)

    def test_temporal_activation_reg(self):
        # make deterministe
        th.manual_seed(0) 
        # some dummy params
        emb_size = 3
        beta = 1
        model = net.AWD_LSTM(10, emb_size, 2)
        # manually set the output to some random values
        # in practice this is done automatically on each forward pass
        # expects model output of shape (timesteps, batch_size, embedding_size)
        # batch_size = 2
        # timesteps = 2
        model.output_nodrop = T([[[3, 4, 5],
                                  [3, 4, 5]],
                                 [[1, 1, 1],
                                  [1, 1, 1]]])
    
        # We expect the L2 norm of the difference between the two timesteps
        # L2(ht - ht+1)
        expect = (T([[2, 3, 4]]) @ T([[2, 3, 4]]).t()).pow(0.5)
        tar = model.temporal_activation_reg(beta)
        self.assertEqual(tar, expect)

    # TODO: Fix these tests to account for timestep!!
    # def test_VariationalDropout(self):
    #     # Fix random seed to make test deterministic
    #     # otherwise there is a non-zero probability 
    #     # of random masks matching
    #     th.manual_seed(0)
    #     # Test
    #     x1 = th.ones(2, 5)
    #     x2 = th.ones(2, 5)
    #     varidrop = net.VariationalDropout(x1, p=0.5)
    #     y1 = varidrop(x1)
    #     y2 = varidrop(x2)
    #     matching_elements = (y1.flatten() == y2.flatten()).sum()
    #     total_elements = y1.flatten().size(0)

    #     # Has dropout been applied?
    #     # The input was matrix of ones, so if sum of elements
    #     # is not total number of elements then we assume some
    #     # elements have been set to zero
    #     self.assertTrue(y1.sum() != total_elements)
    #     # Has the same dropout mask been applied in both calls?
    #     # If true then all elements should match in this setting.
    #     self.assertEqual(matching_elements, total_elements)

    # def test_VariationalDropout_MiniBatch(self):
    #     # Fix random seed to make test deterministic
    #     # otherwise there is a non-zero probability 
    #     # of random masks matching
    #     th.manual_seed(0)
    #     # Test
    #     x1 = th.ones(2, 2, 5)
    #     x2 = th.ones(2, 2, 5)
    #     varidrop = net.VariationalDropout(x1, p=0.5)
    #     y1 = varidrop(x1)
    #     y2 = varidrop(x2)

    #     # Has dropout been applied?
    #     # The input was matrix of ones, so if sum of elements
    #     # is not total number of elements then we assume some
    #     # elements have been set to zero
    #     total_elements = y1.flatten().size(0)
    #     self.assertTrue(y1.sum() != total_elements)
    #     # Has the same dropout mask been applied in both calls?
    #     # If true then all elements should match in this setting.
    #     matching_elements = (y1.flatten() == y2.flatten()).sum()
    #     self.assertEqual(matching_elements, total_elements)
    #     # Is dropout mask different between batches?
    #     # We want it to be according to AWD-LSTM paper:
    #     # > "Each example within the mini-batch uses a unique 
    #     # > dropout mask, rather than a single dropout mask being 
    #     # > used over all examples, ensuring diversity in the 
    #     # > elements dropped out."
    #     matches_between_minibatches = (y1[0].flatten() == y2[1].flatten()).sum()
    #     total_elements = y1[0].flatten().size(0)
    #     self.assertTrue(matches_between_minibatches != total_elements)


suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
unittest.TextTestRunner(verbosity=2).run(suite)