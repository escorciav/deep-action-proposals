import unittest

import pandas as pd

from activitynet_helper import ActivityNet


class test_activitynet(unittest.TestCase):
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
