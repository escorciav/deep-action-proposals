import unittest

import numpy as np

from daps.utils.extra import uniform_batches


class GeneralUtilities(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_balance_labels(self):
        pass

    @unittest.skip("A contribution is required")
    def test_idx_of_queries(self):
        pass

    def test_uniform_batches(self):
        batch_size = 4
        x = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        rst, rst_idx = uniform_batches(x, batch_size)
        self.assertEqual(x.size, rst.size)
        self.assertEqual(rst_idx.size, rst_idx.size)
        for i in range(x.size / batch_size):
            mini_batch = rst[i * batch_size:(i + 1) * batch_size]
            self.assertGreater(mini_batch.sum(), 0)

        # Shrink
        rst, rst_idx = uniform_batches(x, batch_size, return_all=False)
        self.assertGreater(x.size, rst.size)
        self.assertEqual(rst.size, rst_idx.size)

        # Not enough number of positives
        y = np.hstack([x[:, np.newaxis], np.zeros((x.size, 1), dtype=int)])
        batch_size = 2
        rst, rst_idx = uniform_batches(y, batch_size)
        self.assertEqual(rst.shape[0], rst_idx.size)
        for i in range(x.size / batch_size):
            mini_batch = rst[i * batch_size:(i + 1) * batch_size]
            self.assertGreater(mini_batch.sum(), 0)


class SamplingUtilities(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_sampling_with_uniform_groups(self):
        pass


class StringUtilities(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def levenshtein_distance(self):
        pass


class IOUtilities(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_dump_json(self):
        pass

    @unittest.skip("A contribution is required")
    def test_file_as_folder(self):
        pass
