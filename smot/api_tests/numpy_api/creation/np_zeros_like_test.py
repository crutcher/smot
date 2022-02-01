import unittest

import numpy as np

from smot.doc_link.link_annotations import api_link
from smot.testlib import np_eggs


@api_link(
    target="numpy.zeros_like",
    ref="https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html",
)
class ZerosLikeTest(unittest.TestCase):
    def test_zeros_like(self) -> None:
        # Dense ndarrays
        for dtype in [int, float]:
            source = np.ndarray(  # type: ignore
                [1, 2],
                dtype=dtype,
            )

            np_eggs.assert_ndarray_equals(
                np.zeros_like(source),
                np.zeros(
                    source.shape,
                    dtype=source.dtype,
                ),
            )
