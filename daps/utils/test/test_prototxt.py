import unittest

import daps.utils.prototxt as prototxt


class TestMultilineStrings(unittest.TestCase):
    def test_c3d_model(self):
        arguments = {'seq_source': 'hola', 'mean_file': 'mundo',
                     'batch_size': 50, 'use_image': 'false'}
        mlstr = prototxt.c3d_model.format(**arguments)
        self.assertIsInstance(mlstr, str)
