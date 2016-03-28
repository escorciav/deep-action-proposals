import unittest

import numpy as np
import pandas as pd

import baseline as model


class TestBaselineData(unittest.TestCase):
    def setUp(self):
        self.filename = 'data/thumos14/metadata/val_segments_list.txt'
        self.model = model.BaselineData.fromcsv(self.filename)

    def test_from_csv(self):
        filename = 'nonexistent'
        self.assertRaises(IOError, model.BaselineData.fromcsv, filename)
        self.assertIsInstance(self.model.data, pd.DataFrame)
        # TODO: include test with wc -l filename match number of lines

    def test_get_temporal_loc(self):
        self.assertEqual(2, self.model.get_temporal_loc().shape[1])


class TestUtils(unittest.TestCase):
    def test_proposals_per_video(self):
        self.assertRaises(ValueError, model.proposals_per_video,
                          np.random.randn(1))
        x = np.random.randn(2000, 2)
        self.assertRaises(ValueError, model.proposals_per_video, x)
        y = model.proposals_per_video(x, 200)
        self.assertEqual((10, 200, 2), y.shape)
        for i in range(10):
            np.testing.assert_array_equal(x[i*200:(i+1)*200, :], y[i, :, :])
        y = model.proposals_per_video(x, n_videos=20)
        self.assertEqual((20, 100, 2), y.shape)
        for i in range(20):
            np.testing.assert_array_equal(x[i*100:(i+1)*100, :], y[i, :, :])
