import unittest

import numpy as np

from smot.testlib import np_eggs


class EyeTest(unittest.TestCase):
    # https://numpy.org/doc/stable/reference/generated/numpy.eye.html

    def test_eye_zero(self):
        # eye(0) still returns a (0,0) ndarray.
        np_eggs.assert_ndarray(
            np.eye(0),
            np.ones((0, 0)),
        )

    def test_eye(self):
        np_eggs.assert_ndarray(
            np.eye(3),
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

        np_eggs.assert_ndarray(
            np.eye(3, 2),
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
        )

        np_eggs.assert_ndarray(
            np.eye(3, 4),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        )
