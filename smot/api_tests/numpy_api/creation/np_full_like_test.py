import unittest

import numpy as np

from smot.doc_link.link_annotations import api_link
from smot.testlib import np_eggs


@api_link(
    target="numpy.full_like",
    ref="https://numpy.org/doc/stable/reference/generated/numpy.full_like.html",
)
class FullLikeTest(unittest.TestCase):
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
