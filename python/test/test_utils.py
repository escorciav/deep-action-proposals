import os
import shutil
import tempfile
import unittest
from subprocess import check_output

import numpy as np

import utils
from utils import c3d_input_file_generator


class test_c3d_utilities(unittest.TestCase):
    def test_c3d_input_file_generator(self):
        filename = 'not_existent_file'
        self.assertRaises(ValueError, c3d_input_file_generator, filename, '')

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write('video-name num-frame i-frame duration label\n'
                    'my_video 50 0 22 0\nmy_video 50 10 33 1\n'
                    'myvideo2 30 0 15 2\n')
            filename = f.name
            file_in_out = [filename + '.in', filename + '.out']
            dir_out = filename + '_dir'
            summary = utils.c3d_input_file_generator(filename, file_in_out,
                                                     output_folder=dir_out)
            self.assertTrue(summary['success'])
            self.assertEqual(1.0/3, summary['pctg-skipped-segments'])
            self.assertEqual(4.0/3, summary['ratio-clips-segments'])
            # dummy test to double check that output has the right number of
            # clips
            self.assertTrue(os.path.isfile(filename + '.in'))
            rst = check_output(['wc', '-l', filename + '.in']).split(' ')[0]
            self.assertEqual('4', rst)
            self.assertTrue(os.path.isfile(filename + '.out'))
            rst = check_output(['wc', '-l', filename + '.out']).split(' ')[0]
            self.assertEqual('4', rst)
            self.assertTrue(os.path.isdir(filename + '_dir'))
        os.remove(filename)
        os.remove(filename + '.in')
        os.remove(filename + '.out')
        shutil.rmtree(filename + '_dir')

    @unittest.skip("A contribution is required")
    def c3d_read_feature(self):
        pass

    @unittest.skip("A contribution is required")
    def test_c3d_stack_feature(self):
        pass


class test_general_utilities(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_idx_of_queries(self):
        pass


def test_count_frames():
    filename = 'not_existent_video.avi'
    assert utils.count_frames(filename) == 0
    filename = 'data/videos/example.mp4'
    assert utils.count_frames(filename) == 1507
    assert utils.count_frames(filename, 'ffprobe') == 1507
    assert utils.count_frames(os.path.splitext(filename)[0],
                              method='dir') == 1507


def test_frame_rate():
    assert isinstance(utils.frame_rate('data/videos/examples.mp4'), float)
    assert utils.frame_rate('nonexistent.video') == 0.0


def test_video_duration():
    assert isinstance(utils.video_duration('data/videos/examples.mp4'), float)
    assert utils.video_duration('nonexistent.video') == 0.0


class test_sampling_utilities(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_sampling_with_uniform_groups(self):
        pass


class test_segment_utilities(unittest.TestCase):
    def test_intersection(self):
        a = np.random.rand(1)
        b = np.array([[1, 10], [5, 20], [16, 25]])
        self.assertRaises(ValueError, utils.segment_intersection, a, b)
        a = np.random.rand(100, 2)
        self.assertEqual((100, 3, 2), utils.segment_intersection(a, b).shape)
        a = np.array([[5, 15]])
        gt_isegs = np.array([[[5, 10], [5, 15], [16, 15]]], dtype=float)
        np.testing.assert_array_equal(gt_isegs,
                                      utils.segment_intersection(a, b))
        results = utils.segment_intersection(a, b, True)
        self.assertEqual(2, len(results))
        self.assertEqual((a.shape[0], b.shape[0]), results[1].shape)

    def test_iou(self):
        a = np.array([[1, 10], [5, 20], [16, 25]])
        b = np.random.rand(1)
        self.assertRaises(ValueError, utils.segment_iou, a, b)
        b = np.random.rand(100, 2)
        self.assertEqual((3, 100), utils.segment_iou(a, b).shape)
        b = np.array([[1, 10], [1, 30], [10, 20], [20, 30]])
        rst = utils.segment_iou(a, b)
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
        self.assertRaises(ValueError, utils.segment_unit_scaling, a, 2)
        size = (3, 2)
        a = np.random.rand(*size)
        rst = utils.segment_unit_scaling(a, 2)
        self.assertEqual(size, rst.shape)
        b = np.random.rand(size[1])
        self.assertRaises(ValueError, utils.segment_unit_scaling, a, 2, b)
        b = np.random.rand(size[0])
        rst = utils.segment_unit_scaling(a, 2, b)
        self.assertTrue(np.may_share_memory(a, rst))
        rst = utils.segment_unit_scaling(a, 2, b, True)
        self.assertFalse(np.may_share_memory(a, rst))


class test_video_utilities(unittest.TestCase):
    def setUp(self):
        self.video = 'data/videos/example.mp4'
        self.video_dir = 'data/videos/example'

    @unittest.skip("Skipping until correct installation of OpenCV")
    def test_dump_video(self):
        with tempfile.NamedTemporaryFile() as f:
            filename = f.name
        clip = utils.get_clip(self.video, 3, 30)
        utils.dump_video(filename, clip)
        self.assertTrue(os.path.isfile(filename))
        self.assertTrue(utils.count_frames(filename) > 0)
        os.remove(filename)
