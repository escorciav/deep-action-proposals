import os
import tempfile
import unittest

import numpy as np

import utilities


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


def test_count_frames():
    filename = 'not_existent_video.avi'
    assert utilities.count_frames(filename) == 0
    filename = 'data/videos/example.mp4'
    assert utilities.count_frames(filename) == 1507
    assert utilities.count_frames(filename, 'ffprobe') == 1507
    assert utilities.count_frames(os.path.splitext(filename)[0],
                                  method='dir') == 1507


def test_frame_rate():
    assert isinstance(utilities.frame_rate('data/videos/examples.mp4'), float)
    assert utilities.frame_rate('nonexistent.video') == 0.0


def test_video_duration():
    assert isinstance(utilities.video_duration('data/videos/examples.mp4'),
                      float)
    assert utilities.video_duration('nonexistent.video') == 0.0


class test_sampling_utilities(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_sampling_with_uniform_groups(self):
        pass


class test_segment_utilities(unittest.TestCase):
    def test_intersection(self):
        a = np.random.rand(1)
        b = np.array([[1, 10], [5, 20], [16, 25]])
        self.assertRaises(ValueError, utilities.segment_intersection, a, b)
        a = np.random.rand(100, 2)
        self.assertEqual((100, 3, 2),
                         utilities.segment_intersection(a, b).shape)
        a = np.array([[5, 15]])
        gt_isegs = np.array([[[5, 10], [5, 15], [16, 15]]], dtype=float)
        np.testing.assert_array_equal(gt_isegs,
                                      utilities.segment_intersection(a, b))
        results = utilities.segment_intersection(a, b, True)
        self.assertEqual(2, len(results))
        self.assertEqual((a.shape[0], b.shape[0]), results[1].shape)

    def test_iou(self):
        a = np.array([[1, 10], [5, 20], [16, 25]])
        b = np.random.rand(1)
        self.assertRaises(ValueError, utilities.segment_iou, a, b)
        b = np.random.rand(100, 2)
        self.assertEqual((3, 100), utilities.segment_iou(a, b).shape)
        b = np.array([[1, 10], [1, 30], [10, 20], [20, 30]])
        rst = utilities.segment_iou(a, b)
        # segment is equal
        self.assertEqual(1.0, rst[0, 0])
        # segment is disjoined
        self.assertEqual(0.0, rst[0, 3])
        # segment is contained
        self.assertEqual(10.0/30, rst[2, 1])
        # segment to left
        self.assertEqual(5.0/16, rst[2, 2])
        # segment to right
        self.assertEqual(6/15.0, rst[2, 3])

    def test_unit_scaling(self):
        a = np.random.rand(1)
        self.assertRaises(ValueError, utilities.segment_unit_scaling, a, 2)
        size = (3, 2)
        a = np.random.rand(*size)
        rst = utilities.segment_unit_scaling(a, 2)
        self.assertEqual(size, rst.shape)
        b = np.random.rand(size[1])
        self.assertRaises(ValueError, utilities.segment_unit_scaling, a, 2, b)
        b = np.random.rand(size[0])
        rst = utilities.segment_unit_scaling(a, 2, b)
        self.assertTrue(np.may_share_memory(a, rst))
        rst = utilities.segment_unit_scaling(a, 2, b, True)
        self.assertFalse(np.may_share_memory(a, rst))

    def test_segment_format(self):
        a = np.array([[10, 3]])
        res = np.array([[10, 12]])
        np.testing.assert_array_equal(utilities.segment_format(a, 'd2b'), res)


class test_video_utilities(unittest.TestCase):
    def setUp(self):
        self.video = 'data/videos/example.mp4'
        self.video_dir = 'data/videos/example'

    @unittest.skip("Skipping until correct installation of OpenCV")
    def test_dump_video(self):
        with tempfile.NamedTemporaryFile() as f:
            filename = f.name
        clip = utilities.get_clip(self.video, 3, 30)
        utilities.dump_video(filename, clip)
        self.assertTrue(os.path.isfile(filename))
        self.assertTrue(utilities.count_frames(filename) > 0)
        os.remove(filename)
