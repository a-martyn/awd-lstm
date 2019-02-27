import unittest
import string
import torch as th
import torch.nn as nn

# Local
from utils import NT_ASGD


class Tests(unittest.TestCase):
    
    def test_NT_ASGD(self):
        model = nn.LSTM(10, 20, 1)
        lr = 0.4
        weight_decay = 0.1
        n = 3
        
        nt_asgd = NT_ASGD(lr, weight_decay, n)
        # ASGD used in place of SGD optimizer when
        # loss increases for n succesive calls to get_optimizer 
        self.assertEqual(nt_asgd.asgd_triggered, False)
        nt_asgd.get_optimizer(3)
        self.assertEqual(nt_asgd.asgd_triggered, False)
        nt_asgd.get_optimizer(2)
        self.assertEqual(nt_asgd.asgd_triggered, False)
        nt_asgd.get_optimizer(3)
        self.assertEqual(nt_asgd.asgd_triggered, False)
        nt_asgd.get_optimizer(3)
        self.assertEqual(nt_asgd.asgd_triggered, False)
        # ASGD Triggered because loss was lowest n+1 epochs ago
        nt_asgd.get_optimizer(4)
        self.assertEqual(nt_asgd.asgd_triggered, True)
        nt_asgd.get_optimizer(2)
        self.assertEqual(nt_asgd.asgd_triggered, True)
        nt_asgd.get_optimizer(3)
        self.assertEqual(nt_asgd.asgd_triggered, True)
        nt_asgd.get_optimizer(3)
        self.assertEqual(nt_asgd.asgd_triggered, True) 
        # Doesn't un-trigger
        nt_asgd.get_optimizer(4)
        self.assertEqual(nt_asgd.asgd_triggered, True)  
        


suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
unittest.TextTestRunner(verbosity=2).run(suite)