import os
import shutil
import tempfile
from subprocess import check_output

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
