import unittest

import nose.tools as nt
import numpy as np

from daps.c3d_feature_helper import pyramid1d


class TestFeature(unittest.TestCase):
    @unittest.skip("A contribution is required")
    def test_read_feat():
        pass

    @unittest.skip("A contribution is required")
    def test_read_feat_batch_from_video():
        pass


@unittest.skip("A contribution is required")
def test_concat1d():
    return None


def test_pyramid1d():
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
    nt.assert_equal((2,), pyramid1d(x).shape)
    py1_x = pyramid1d(x, 1)
    nt.assert_equal((6,), py1_x.shape)
    rst = np.array([2, 3])/np.sqrt(13)
    np.testing.assert_array_almost_equal(rst, py1_x[4:6])
    py2_x = pyramid1d(x, 2)
    nt.assert_equal((14,), py2_x.shape)
    rst = np.array([1, 2])/np.sqrt(5)
    np.testing.assert_array_almost_equal(rst, py2_x[8:10])
