import unittest

import numpy as np

from smot.testlib import np_eggs


class FullTest(unittest.TestCase):
    # https://numpy.org/doc/stable/reference/generated/numpy.full.html

    def test_full_scalar(self):
        np_eggs.assert_ndarray(
            np.full(tuple(), 2.0),
            np.array(2.0, dtype=float),
        )

    def test_full(self):
        for dtype in [np.int8, np.float32]:
            np_eggs.assert_ndarray(
                np.full((3,), 2, dtype=dtype),
                np.array([2, 2, 2], dtype=dtype),
            )
