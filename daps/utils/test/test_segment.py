import unittest

import numpy as np

import daps.utils.segment as segment


class test_segment_utilities(unittest.TestCase):
    def test_format(self):
        a = np.array([[10, 3]])
        res = np.array([[10, 12]])
        np.testing.assert_array_equal(segment.format(a, 'd2b'), res)

    def test_intersection(self):
        a = np.random.rand(1)
        b = np.array([[1, 10], [5, 20], [16, 25]])
        self.assertRaises(ValueError, segment.intersection, a, b)
        a = np.random.rand(100, 2)
        self.assertEqual((100, 3, 2), segment.intersection(a, b).shape)
        a = np.array([[5, 15]])
        gt_isegs = np.array([[[5, 10], [5, 15], [16, 15]]], dtype=float)
        np.testing.assert_array_equal(gt_isegs, segment.intersection(a, b))
        results = segment.intersection(a, b, True)
        self.assertEqual(2, len(results))
        self.assertEqual((a.shape[0], b.shape[0]), results[1].shape)

    def test_iou(self):
        a = np.array([[1, 10], [5, 20], [16, 25]])
        b = np.random.rand(1)
        self.assertRaises(ValueError, segment.iou, a, b)
        b = np.random.rand(100, 2)
        self.assertEqual((3, 100), segment.iou(a, b).shape)
        b = np.array([[1, 10], [1, 30], [10, 20], [20, 30]])
        rst = segment.iou(a, b)
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

    @unittest.skip("A contribution is required")
    def test_nms_detection(self):
        pass

    def test_unit_scaling(self):
        a = np.random.rand(1)
        self.assertRaises(ValueError, segment.unit_scaling, a, 2)
        size = (3, 2)
        a = np.random.rand(*size)
        rst = segment.unit_scaling(a, 2)
        self.assertEqual(size, rst.shape)
        b = np.random.rand(size[1])
        self.assertRaises(ValueError, segment.unit_scaling, a, 2, b)
        b = np.random.rand(size[0])
        rst = segment.unit_scaling(a, 2, b)
        self.assertTrue(np.may_share_memory(a, rst))
        rst = segment.unit_scaling(a, 2, b, True)
        self.assertFalse(np.may_share_memory(a, rst))
