import unittest

import numpy as np

from smot.doc_link.link_annotations import api_link
from smot.testlib import np_eggs


@api_link(
    target="numpy.full",
    ref="https://numpy.org/doc/stable/reference/generated/numpy.full.html",
)
class FullTest(unittest.TestCase):
    def test_full_scalar(self) -> None:
        np_eggs.assert_ndarray(
            np.full(tuple(), 2.0),
            np.array(2.0, dtype=float),
        )

    def test_full(self) -> None:
        for dtype in [np.int8, np.float32]:
            np_eggs.assert_ndarray(
                np.full((3,), 2, dtype=dtype),
                np.array([2, 2, 2], dtype=dtype),
            )
