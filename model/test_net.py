import unittest
import string
import torch as th

# Local
import net


class Tests(unittest.TestCase):

    def test_drop_connect(self):
        # Instantiate an LSTM net with weight dropout on 
        # hidden-to-hidden weights
        model = net.AWD_LSTM(10, 20, 2)
        old_state = model.state_dict()

        # Apply drop connect with probabilit 1.0
        model.drop_connect(p=1.0) 
        new_state = model.state_dict()
        
        # Test
        # Because dropout probability is 1
        # all hidden-to-hidden weights should be zero
        # There are two such weight matrices because we
        # configured a 2-layer lstm
        self.assertTrue(new_state['layer0.h2h.weight'].sum() == 0)
        self.assertTrue(new_state['layer1.h2h.weight'].sum() == 0)
        # Here we test that input-to-hidden weights are unaffected
        self.assertTrue(new_state['layer0.i2h.weight'].sum() != 0)
        self.assertTrue(new_state['layer1.i2h.weight'].sum() != 0)


    
     # TODO: Fix these tests to account for timestep!!
    def test_VariationalDropout(self):
        # Fix random seed to make test deterministic
        # otherwise there is a non-zero probability 
        # of random masks matching
        th.manual_seed(0)
        # Test
        x1 = th.ones(2, 5)
        x2 = th.ones(2, 5)
        varidrop = net.VariationalDropout(x1, p=0.5)
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
        th.manual_seed(0)
        # Test
        x1 = th.ones(2, 2, 5)
        x2 = th.ones(2, 2, 5)
        varidrop = net.VariationalDropout(x1, p=0.5)
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