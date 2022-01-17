import unittest

import numpy as np

from smot.testlib import np_eggs


class IdentityTest(unittest.TestCase):
    # https://numpy.org/doc/stable/reference/generated/numpy.identity.html

    def test_identity_zero(self) -> None:
        # identity(0) still returns a (0,0) ndarray.
        np_eggs.assert_ndarray(
            np.identity(0),
            np.ones((0, 0)),
        )

    def test_identity(self) -> None:
        np_eggs.assert_ndarray(
            np.identity(3),
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
