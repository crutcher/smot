import unittest

import numpy as np

from smot.doc_link.link_annotations import api_link
from smot.testlib import np_eggs


@api_link(
    target="numpy.ones_like",
    ref="https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html",
)
class OnesLikeTest(unittest.TestCase):
    def test_ones_like(self) -> None:
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
