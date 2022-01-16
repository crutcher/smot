import unittest

import numpy as np

from smot.testlib import np_eggs


class EmptyLikeTest(unittest.TestCase):
    def test_empty_like_scalar(self):
        a = np.array(0)

        np_eggs.assert_ndarray_structure(
            np.empty_like(a),
            a,
        )

    def test_empty_like(self):
        for dtype in [int, float]:
            for data in [0, [[0]], [[1], [2]]]:
                t = np.array(data, dtype=dtype)  # type: ignore

                np_eggs.assert_ndarray_structure(
                    np.empty_like(t),
                    t,
                )
