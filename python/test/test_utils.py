import os
import shutil
import tempfile
import unittest
from subprocess import check_output

import numpy as np

import utils


def test_c3d_input_file_generator():
    filename = 'not_existent_file'
    summary = utils.c3d_input_file_generator(filename, '')
    assert summary['success'] == False

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write('my_video 50 0 22 0\nmy_video 50 10 33 1\n' +
                'myvideo2 30 0 15 2\n')
        filename = f.name
    summary = utils.c3d_input_file_generator(filename, [filename + '.in',
                                                        filename + '.out'],
                                             output_folder=filename + '_dir')
    assert summary['success'] == True
    assert summary['pctg-skipped-segments'] == (1.0/3)
    assert summary['ratio-clips-segments'] == (4.0/3)
    # dummy test to double check that output has the right number of clips
    assert os.path.isfile(filename + '.in') == True
    assert check_output(['wc', '-l', filename + '.in']).split(' ')[0] == '4'
    assert os.path.isfile(filename + '.out') == True
    assert check_output(['wc', '-l', filename + '.out']).split(' ')[0] == '4'
    assert os.path.isdir(filename + '_dir') == True
    os.remove(filename)
    os.remove(filename + '.in')
    os.remove(filename + '.out')
    shutil.rmtree(filename + '_dir')
    return None

# Tests of general purpose functions


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


class test_segment_utilities(unittest.TestCase):
    def test_segment_intersection(self):
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
