import unittest

import numpy as np

import daps.utilities as utilities


class test_general_utilities(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_idx_of_queries(self):
        pass

    def test_feature_1dpyramid(self):
        x = np.array([[0, 4],
                      [4, 2],
                      [0, 4],
                      [2, 0],
                      [1, 2],
                      [1, 4],
                      [3, 4],
                      [1, 4],
                      [1, 1],
                      [4, 2]])
        self.assertEqual((2,), utilities.feature_1dpyramid(x).shape)
        py1_x = utilities.feature_1dpyramid(x, 1)
        self.assertEqual((6,), py1_x.shape)
        rst = np.array([2, 3])/np.sqrt(13)
        np.testing.assert_array_almost_equal(rst, py1_x[4:6])
        py2_x = utilities.feature_1dpyramid(x, 2)
        self.assertEqual((14,), py2_x.shape)
        rst = np.array([1, 2])/np.sqrt(5)
        np.testing.assert_array_almost_equal(rst, py2_x[8:10])

    def test_uniform_batches(self):
        batch_size = 4
        x = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        rst, rst_idx = utilities.uniform_batches(x, batch_size)
        self.assertEqual(x.size, rst.size)
        self.assertEqual(rst_idx.size, rst_idx.size)
        for i in range(x.size / batch_size):
            mini_batch = rst[i * batch_size:(i + 1) * batch_size]
            self.assertGreater(mini_batch.sum(), 0)

        # Shrink
        rst, rst_idx = utilities.uniform_batches(x, batch_size,
                                                 return_all=False)
        self.assertGreater(x.size, rst.size)
        self.assertEqual(rst.size, rst_idx.size)

        # Not enough number of positives
        y = np.hstack([x[:, np.newaxis], np.zeros((x.size, 1), dtype=int)])
        batch_size = 2
        rst, rst_idx = utilities.uniform_batches(y, batch_size)
        self.assertEqual(rst.shape[0], rst_idx.size)
        for i in range(x.size / batch_size):
            mini_batch = rst[i * batch_size:(i + 1) * batch_size]
            self.assertGreater(mini_batch.sum(), 0)


class test_sampling_utilities(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_sampling_with_uniform_groups(self):
        pass
