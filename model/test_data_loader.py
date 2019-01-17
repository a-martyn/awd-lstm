import unittest
import string
import torch as th
import numpy as np

# Local
import data_loader as dl


class Tests(unittest.TestCase):

    def test_batch_A(self):
        """batch(): with truncation"""
        d = th.LongTensor(list(range(26)))
        batch_size = 4
        expect = th.LongTensor([[0, 6, 12, 18],
                                [1, 7, 13, 19],
                                [2, 8, 14, 20],
                                [3, 9, 15, 21],
                                [4, 10, 16, 22],
                                [5, 11, 17, 23]])
        r = dl.batch(d, batch_size)
        self.assertEqual(int(th.all(th.eq(expect, r))), 1)

    def test_batch_B(self):
        """batch(): with remainder zero"""
        d = th.LongTensor(list(range(24)))
        batch_size = 4
        expect = th.LongTensor([[0, 6, 12, 18],
                                [1, 7, 13, 19],
                                [2, 8, 14, 20],
                                [3, 9, 15, 21],
                                [4, 10, 16, 22],
                                [5, 11, 17, 23]])
        r = dl.batch(d, batch_size)
        self.assertEqual(int(th.all(th.eq(expect, r))), 1)


    def test_get_batch_A(self):
        d = th.LongTensor([[0, 6, 12, 18],
                           [1, 7, 13, 19],
                           [2, 8, 14, 20],
                           [3, 9, 15, 21],
                           [4, 10, 16, 22],
                           [5, 11, 17, 23]])
        i = 0
        timesteps = 2
        xe = th.LongTensor([[0, 6, 12, 18],
                               [1, 7, 13, 19]])
        ye = th.LongTensor([[1, 7, 13, 19],
                               [2, 8, 14, 20]])
        x, y, _ = dl.get_batch(d, i, timesteps)

        self.assertEqual(int(th.all(th.eq(xe, x))), 1)
        self.assertEqual(int(th.all(th.eq(ye, y))), 1)

    def test_get_batch_B(self):
        d = th.LongTensor([[0, 6, 12, 18],
                           [1, 7, 13, 19],
                           [2, 8, 14, 20],
                           [3, 9, 15, 21],
                           [4, 10, 16, 22],
                           [5, 11, 17, 23]])
        i = 4
        timesteps = 2
        xe = th.LongTensor([[4, 10, 16, 22]])
        ye = th.LongTensor([[5, 11, 17, 23]])
        x, y, _ = dl.get_batch(d, i, timesteps)

        self.assertEqual(int(th.all(th.eq(xe, x))), 1)
        self.assertEqual(int(th.all(th.eq(ye, y))), 1)

    def test_get_batch_jitter(self):
        """When jitter is true, sequence length becomes variable on each call"""
        th.manual_seed(1)
        np.random.seed(0)
        d = th.randn((50, 4))
        i = 0
        timesteps = 30
        x, y, seq_len = dl.get_batch(d, i, timesteps, jitter=True)

        # With this random seed, sequence length is 33
        # seq length != timesteps
        self.assertEqual(seq_len, 33)
        self.assertEqual(x.size(0), seq_len)
        self.assertTrue(timesteps != seq_len)

suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
unittest.TextTestRunner(verbosity=2).run(suite)
