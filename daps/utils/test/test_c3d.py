import os
import shutil
import tempfile
import unittest
from subprocess import check_output

from daps.utils.c3d import input_file_generator


class test_c3d_utilities(unittest.TestCase):
    def test_input_file_generator(self):
        filename = 'not_existent_file'
        self.assertRaises(ValueError, input_file_generator, filename, '')

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write('video-name num-frame i-frame duration label\n'
                    'my_video 50 0 22 0\nmy_video 50 10 33 1\n'
                    'myvideo2 30 0 15 2\n')
        filename = f.name
        file_in_out = [filename + '.in', filename + '.out']
        dir_out = filename + '_dir'
        summary = input_file_generator(filename, file_in_out,
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
    def test_read_feature(self):
        pass
