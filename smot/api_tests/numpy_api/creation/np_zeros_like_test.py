import unittest

import numpy as np

from smot.testlib import np_eggs


class ZerosLikeTest(unittest.TestCase):
    # https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html

    def test_zeros_like(self):
        # Dense ndarrays
        for dtype in [int, float]:
            source = np.ndarray(  # type: ignore
                [1, 2],
                dtype=dtype,
            )

            np_eggs.assert_ndarray(
                np.zeros_like(source),
                np.zeros(
                    source.shape,
                    dtype=source.dtype,
                ),
            )
