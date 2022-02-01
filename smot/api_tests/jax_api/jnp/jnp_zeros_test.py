import unittest

import jax.numpy as jnp

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, np_eggs


@api_link(
    target="jax.numpy.zeros",
    ref="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.zeros.html",
)
class ZerosTest(unittest.TestCase):
    def test_default(self) -> None:
        t = jnp.zeros((1, 2))

        np_eggs.assert_ndarray_equals(
            t,
            jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        )

    def test_scalar(self) -> None:
        # np.zeros(size) doesn't have a default;
        # but you can still construct a scalar.
        t = jnp.zeros([])

        eggs.assert_match(t.shape, tuple())
        eggs.assert_match(t.size, 1)
        eggs.assert_match(t.item(), 0)
