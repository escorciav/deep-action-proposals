import os
import unittest

import pandas as pd

from thumos14_helper import Thumos14


class test_thumos14(unittest.TestCase):
    def setUp(self):
        self.assertRaises(IOError, Thumos14, 'nonexistent')
        self.thumos = Thumos14()

    def test_annotation_files(self):
        # Dummy test to verify correct number of files. An exhaustive test is
        # welcome
        self.assertEqual(21, len(self.thumos.annotation_files()))
        self.assertEqual(21, len(self.thumos.annotation_files('test')))

    def test_segments_info(self):
        filename = 'new_file'
        result = self.thumos.segments_info('test', filename)
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(7, result.shape[1])
        self.assertTrue(os.path.isfile(filename))
        os.remove(filename)

    def test_index_from_filename(self):
        actions = ['SoccerPenalty_val.txt', 'b/Ambiguous_val.txt']
        idx = [16, -1]
        for i, v in enumerate(actions):
            self.assertEqual(idx[i], self.thumos.index_from_filename(v))
