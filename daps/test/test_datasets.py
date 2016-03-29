import unittest

import nose.tools as nt
import pandas as pd

from daps.datasets import Dataset, DatasetBase, ActivityNet, Thumos14


def test_Dataset():
    for i in ['thumos14', 'activitynet']:
        ds = Dataset(i)
        nt.assert_is_instance(ds.wrapped_dataset, DatasetBase)
        # Assert main methods
        nt.assert_true(hasattr(ds.wrapped_dataset, 'segments_info'))
        nt.assert_true(hasattr(ds.wrapped_dataset, 'video_info'))


def test_DatasetBase():
    ds = DatasetBase()
    nt.raises(NotImplementedError, ds.segments_info)
    nt.raises(NotImplementedError, ds.video_info)


class test_ActivityNet(unittest.TestCase):
    def setUp(self):
        self.assertRaises(IOError, ActivityNet, 'nonexistent')
        self.anet = ActivityNet()

    def test_video_info(self):
        df_train = self.anet.video_info('train')
        df_val = self.anet.video_info('val')
        df_test = self.anet.video_info('test')
        self.assertEqual(4819, df_train.shape[0])
        self.assertEqual(2383, df_val.shape[0])
        self.assertEqual(2480, df_test.shape[0])

    def test_segments_info(self):
        df_train = self.anet.segments_info('train')
        df_val = self.anet.segments_info('val')
        self.assertEqual(7151, df_train.shape[0])
        self.assertEqual(3582, df_val.shape[0])

    def test_index_from_action_name(self):
        action = 'Long jump'
        index = 80
        self.assertEqual(index, self.anet.index_from_action_name(action))


class test_Thumos14(unittest.TestCase):
    def setUp(self):
        self.assertRaises(IOError, Thumos14, 'nonexistent')
        self.thumos = Thumos14()

    def test_annotation_files(self):
        # Dummy test to verify correct number of files. An exhaustive test is
        # welcome
        self.assertEqual(21, len(self.thumos.annotation_files()))
        self.assertEqual(21, len(self.thumos.annotation_files('test')))

    def test_index_from_filename(self):
        actions = ['SoccerPenalty_val.txt', 'b/Ambiguous_val.txt']
        idx = [16, -1]
        for i, v in enumerate(actions):
            self.assertEqual(idx[i], self.thumos.index_from_filename(v))

    def test_segments_info(self):
        for i in ['val', 'test']:
            result = self.thumos.segments_info(i)
            self.assertTrue(isinstance(result, pd.DataFrame))
            self.assertEqual(len(self.thumos.fields_segment), result.shape[1])

    def test_video_info(self):
        for i in ['val', 'test']:
            result = self.thumos.video_info(i)
            self.assertTrue(isinstance(result, pd.DataFrame))
            self.assertEqual(len(self.thumos.fields_video), result.shape[1])
