import unittest

import prototxt


class TestMultilineStrings(unittest.TestCase):
    def test_c3d_model(self):
        arguments = {'seq_source': 'hola', 'mean_file': 'mundo',
                     'batch_size': 50}
        mlstr = prototxt.c3d_model.format(**arguments)
        self.assertIsInstance(mlstr, str)
