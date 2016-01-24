import unittest
from thumos14_helper import thumos14


class test_thumos14(unittest.TestCase):
    def setUp(self):
        self.assertRaises(IOError, thumos14, 'nonexistent')
        self.thumos = thumos14()

    def test_dir_videos(self):
        result = self.thumos.dir_videos()
        self.assertEqual('data/thumos14/val_mp4', result)
        result = self.thumos.dir_videos('test')
        self.assertEqual('data/thumos14/test_mp4', result)
