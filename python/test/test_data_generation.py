import unittest

import numpy as np

import data_generation


class TestUtils(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_compute_priors(self):
        pass

    def test_constants(self):
        self.assertIsInstance(data_generation.RATIO_INTERVALS, list)
        self.assertIsInstance(data_generation.REQ_INFO_CP, list)

    @unittest.skip("A contribution is required")
    def test_generate_segments(self):
        #rst = ...
        #self.assertIsInstance(rst, np.ndarray)
        #self.assertEqual(2, rst[2].ndim)
        #self.assertEqual(2, rst[2].shape[1])

        #rst = ...
        #self.assertEqual(3, len(rst))
        #self.assertIsInstance(rst[1], list)
        #for i in [0, 2]:
        #    self.assertIsInstance(rst[i], np.ndarray)
        #self.assertEqual(1, rst[2].ndim)
        #self.assertEqual(rst[0].shape[0], len(rst[1]))
        #self.assertEqual(rst[0].shape[0], rst[2].size)
        pass