import unittest

import numpy as np

from smot.doc_link.link_annotations import api_link
from smot.testlib import np_eggs


@api_link(
    target="numpy.empty_like",
    ref="https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html",
)
class EmptyLikeTest(unittest.TestCase):
    def test_empty_like_scalar(self) -> None:
        a: np.typing.ArrayLike = np.array(0)

        np_eggs.assert_ndarray_structure(
            np.empty_like(a),
            a,
        )

    def test_empty_like(self) -> None:
        for dtype in [int, float]:
            for data in [0, [[0]], [[1], [2]]]:
                t = np.array(data, dtype=dtype)  # type: ignore

                np_eggs.assert_ndarray_structure(
                    np.empty_like(t),
                    t,
                )
