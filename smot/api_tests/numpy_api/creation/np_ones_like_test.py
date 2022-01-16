import unittest

import numpy as np

from smot.testlib import np_eggs


class OnesLikeTest(unittest.TestCase):
    # https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html

    def test_ones_like(self):
        # Dense ndarrays
        for dtype in [int, float]:
            source = np.ndarray(  # type: ignore
                [1, 2],
                dtype=dtype,
            )

            np_eggs.assert_ndarray(
                np.ones_like(source),
                np.ones(
                    source.shape,
                    dtype=source.dtype,
                ),
            )
