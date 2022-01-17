import unittest

import numpy as np

from smot.testlib import eggs, np_eggs


class OnesTest(unittest.TestCase):
    # https://numpy.org/doc/stable/reference/generated/numpy.ones.html

    def test_default(self) -> None:
        t = np.ones((1, 2))

        np_eggs.assert_ndarray(
            t,
            [[1.0, 1.0]],
        )

    def test_scalar(self) -> None:
        # np.ones(size) doesn't have a default;
        # but you can still construct a scalar.
        t = np.ones([])

        eggs.assert_match(t.shape, tuple())
        eggs.assert_match(t.size, 1)
        eggs.assert_match(t.item(), 1)
