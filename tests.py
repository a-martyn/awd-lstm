import unittest
import string
import torch

# Local
import data_handlers as data


class Tests(unittest.TestCase):

    def test_batch_A(self):
        """batch(): with truncation"""
        d = torch.LongTensor(list(range(26)))
        batch_size = 4
        expect = torch.LongTensor([[0, 6, 12, 18],
                                   [1, 7, 13, 19],
                                   [2, 8, 14, 20],
                                   [3, 9, 15, 21],
                                   [4, 10, 16, 22],
                                   [5, 11, 17, 23]])
        r = data.batch(d, batch_size)
        self.assertEqual(int(torch.all(torch.eq(expect, r))), 1)

    def test_batch_B(self):
        """batch(): with remainder zero"""
        d = torch.LongTensor(list(range(24)))
        batch_size = 4
        expect = torch.LongTensor([[0, 6, 12, 18],
                                   [1, 7, 13, 19],
                                   [2, 8, 14, 20],
                                   [3, 9, 15, 21],
                                   [4, 10, 16, 22],
                                   [5, 11, 17, 23]])
        r = data.batch(d, batch_size)
        self.assertEqual(int(torch.all(torch.eq(expect, r))), 1)


    def test_get_batch_A(self):
        d = torch.LongTensor([[0, 6, 12, 18],
                              [1, 7, 13, 19],
                              [2, 8, 14, 20],
                              [3, 9, 15, 21],
                              [4, 10, 16, 22],
                              [5, 11, 17, 23]])
        i = 0
        bptt = 2
        xe = torch.LongTensor([[0, 6, 12, 18],
                               [1, 7, 13, 19]])
        ye = torch.LongTensor([[1, 7, 13, 19],
                               [2, 8, 14, 20]])
        x, y = data.get_batch(d, i, bptt)

        self.assertEqual(int(torch.all(torch.eq(xe, x))), 1)
        self.assertEqual(int(torch.all(torch.eq(ye, y))), 1)

    def test_get_batch_B(self):
        d = torch.LongTensor([[0, 6, 12, 18],
                              [1, 7, 13, 19],
                              [2, 8, 14, 20],
                              [3, 9, 15, 21],
                              [4, 10, 16, 22],
                              [5, 11, 17, 23]])
        i = 4
        bptt = 2
        xe = torch.LongTensor([[4, 10, 16, 22]])
        ye = torch.LongTensor([[5, 11, 17, 23]])
        x, y = data.get_batch(d, i, bptt)

        self.assertEqual(int(torch.all(torch.eq(xe, x))), 1)
        self.assertEqual(int(torch.all(torch.eq(ye, y))), 1)
suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
unittest.TextTestRunner(verbosity=2).run(suite)
