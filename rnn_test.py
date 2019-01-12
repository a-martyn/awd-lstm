import unittest
import string
import torch

# Local
import rnn


class Tests(unittest.TestCase):

    def test_WeightDrop(self):
        # Instantiate a GRU Cell with weight dropout on 
        # hidden-to-hidden weights
        module = torch.nn.GRUCell(2, 2)
        old_state = module.state_dict()
        wd_module = rnn.WeightDrop(module, ['weight_hh'], dropout=1.0)

        # Forward pass
        input_ = torch.randn(3, 2)
        h0 = torch.randn(3, 2)
        h1 = wd_module(input_, h0)

        new_state = module.state_dict()

        # Test
        # Because dropout probability is 1
        # all hidden-to-hidden weights should be zero
        self.assertTrue(new_state['weight_hh'].sum() == 0)
        # Because ['weight_hh'] was specified other weights should be
        # unaffected by dropout. Here we test that
        # input-to-hidden weights are unaffected
        self.assertTrue(new_state['weight_ih'].sum() != 0)


    def test_WeightDropLSTM(self):
        # Instantiate an LSTM net with weight dropout on 
        # hidden-to-hidden weights
        wd_lstm = rnn.WeightDropLSTM(10, 20, 2, weight_dropout=1.0)
        old_state = wd_lstm.state_dict()

        # Forward pass
        input_ = torch.randn(5, 3, 10)
        h0 = torch.randn(2, 3, 20)
        c0 = torch.randn(2, 3, 20)
        h_ = wd_lstm(input_, (h0, c0))

        new_state = wd_lstm.state_dict()
        
        # Test
        # Because dropout probability is 1
        # all hidden-to-hidden weights should be zero
        # There are two such weight matrices because we
        # configured a 2-layer lstm
        self.assertTrue(new_state['weight_hh_l0'].sum() == 0)
        self.assertTrue(new_state['weight_hh_l1'].sum() == 0)
        # Because ['weight_hh'] was specified other weights should be
        # unaffected by dropout. Here we test that
        # input-to-hidden weights are unaffected
        self.assertTrue(new_state['weight_ih_l0'].sum() != 0)
        self.assertTrue(new_state['weight_ih_l1'].sum() != 0)


    def test_VariationalDropout(self):
        # Fix random seed to make test deterministic
        # otherwise there is a non-zero probability 
        # of random masks matching
        torch.manual_seed(0)
        # Test
        x1 = torch.ones(2, 5)
        x2 = torch.ones(2, 5)
        varidrop = rnn.VariationalDropout(x1, p=0.5)
        y1 = varidrop(x1)
        y2 = varidrop(x2)
        matching_elements = (y1.flatten() == y2.flatten()).sum()
        total_elements = y1.flatten().size(0)

        # Has dropout been applied?
        # The input was matrix of ones, so if sum of elements
        # is not total number of elements then we assume some
        # elements have been set to zero
        self.assertTrue(y1.sum() != total_elements)
        # Has the same dropout mask been applied in both calls?
        # If true then all elements should match in this setting.
        self.assertEqual(matching_elements, total_elements)

    def test_VariationalDropout_MiniBatch(self):
        # Fix random seed to make test deterministic
        # otherwise there is a non-zero probability 
        # of random masks matching
        torch.manual_seed(0)
        # Test
        x1 = torch.ones(2, 2, 5)
        x2 = torch.ones(2, 2, 5)
        varidrop = rnn.VariationalDropout(x1, p=0.5)
        y1 = varidrop(x1)
        y2 = varidrop(x2)

        # Has dropout been applied?
        # The input was matrix of ones, so if sum of elements
        # is not total number of elements then we assume some
        # elements have been set to zero
        total_elements = y1.flatten().size(0)
        self.assertTrue(y1.sum() != total_elements)
        # Has the same dropout mask been applied in both calls?
        # If true then all elements should match in this setting.
        matching_elements = (y1.flatten() == y2.flatten()).sum()
        self.assertEqual(matching_elements, total_elements)
        # Is dropout mask different between batches?
        # We want it to be according to AWD-LSTM paper:
        # > "Each example within the mini-batch uses a unique 
        # > dropout mask, rather than a single dropout mask being 
        # > used over all examples, ensuring diversity in the 
        # > elements dropped out."
        matches_between_minibatches = (y1[0].flatten() == y2[1].flatten()).sum()
        total_elements = y1[0].flatten().size(0)
        self.assertTrue(matches_between_minibatches != total_elements)




suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
unittest.TextTestRunner(verbosity=2).run(suite)
