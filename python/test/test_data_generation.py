import unittest

import data_generation


class TestUtils(unittest.TestCase):
    def testConstants(self):
        self.assertIsInstance(data_generation.RATIO_INTERVALS, list)

    @unittest.skip("A contribution is required")
    def test_generate_segments(self):
        pass
