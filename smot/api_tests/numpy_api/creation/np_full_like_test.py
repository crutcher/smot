import unittest

import numpy as np

from smot.testlib import np_eggs


class FullLikeTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.full_like.html

    def test_full_like_scalar(self) -> None:
        src: np.typing.ArrayLike = np.array(0)

        np_eggs.assert_ndarray_structure(
            np.full_like(src, 2),
            np.array(2),
        )

    def test_full_like(self) -> None:
        for dtype in [np.int8, np.float32]:
            src: np.typing.ArrayLike = np.array([1], dtype=dtype)
            np_eggs.assert_ndarray(
                np.full_like(src, 2),
                np.array([2], dtype=dtype),
            )

            src = np.array([[1], [1]], dtype=dtype)
            np_eggs.assert_ndarray(
                np.full_like(src, 2),
                np.array([[2], [2]], dtype=dtype),
            )
