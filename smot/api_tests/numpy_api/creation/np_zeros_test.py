import unittest

import numpy as np

from smot.api_tests.doc_links import api_link
from smot.testlib import eggs, np_eggs


@api_link(
    target="np.zeros",
    ref="https://numpy.org/doc/stable/reference/generated/numpy.zeros.html",
)
class ZerosTest(unittest.TestCase):
    def test_default(self) -> None:
        t = np.zeros((1, 2))

        np_eggs.assert_ndarray(
            t,
            [[0.0, 0.0]],
        )

    def test_scalar(self) -> None:
        # np.zeros(size) doesn't have a default;
        # but you can still construct a scalar.
        t = np.zeros([])

        eggs.assert_match(t.shape, tuple())
        eggs.assert_match(t.size, 1)
        eggs.assert_match(t.item(), 0)
