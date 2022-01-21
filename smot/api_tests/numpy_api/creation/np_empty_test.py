import unittest

import numpy as np

from smot.api_tests.doc_links import api_link
from smot.testlib import np_eggs


@api_link(
    target="np.empty",
    ref="https://numpy.org/doc/stable/reference/generated/numpy.empty.html",
)
class EmptyTest(unittest.TestCase):
    def test_empty_zero(self) -> None:
        np_eggs.assert_ndarray(
            np.empty(0),
            np.ones(0),
        )

    def test_empty(self) -> None:
        np_eggs.assert_ndarray_structure(
            np.empty([2, 2]),
            np.array([[0.0, 0.0], [0.0, 0.0]]),
        )

        for dtype in [int, float]:
            np_eggs.assert_ndarray_structure(
                np.empty(3, dtype=dtype),
                np.array(
                    # random data ...
                    [0, 0, 0],
                    dtype=dtype,
                ),
            )
