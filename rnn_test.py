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

        print(old_state)
        print(new_state)

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



suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
unittest.TextTestRunner(verbosity=2).run(suite)
