import unittest

import jax.random

from smot.examples.jax import jax_sirens


class SirensTest(unittest.TestCase):
    def test_nothing(self) -> None:
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key, 2)
        x = jax.random.uniform(subkey, [8, 2])
        net = jax_sirens.SirenNet(
            num_channels=3,
            d_hidden=256,
            depth=5,
            w0_initial=30,
        )
        params = net.init(key, x)
        c = net.apply(params, x)
        assert c.shape == (8, 3)  # type: ignore
