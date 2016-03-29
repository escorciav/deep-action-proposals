import unittest

import numpy as np
import theano
import theano.tensor as T

from daps.model import weigthed_binary_crossentropy


class test_loss_functions(unittest.TestCase):
    def test_weigthed_binary_crossentropy(self):
        w0_val, w1_val = 0.5, 1.0
        x_val, y_val = np.random.rand(5, 3), np.random.randint(0, 2, (5, 3))
        expected_val = -(w1_val * y_val * np.log(x_val) +
                         w0_val * (1 - y_val) * np.log(1 - x_val))

        w0, w1 = T.constant(w0_val), T.constant(w1_val)
        x, y = T.matrix('pred'), T.matrix('true')
        loss = weigthed_binary_crossentropy(x, y, w0, w1)
        f = theano.function([x, y], loss, allow_input_downcast=True)

        np.testing.assert_array_almost_equal(expected_val, f(x_val, y_val))
