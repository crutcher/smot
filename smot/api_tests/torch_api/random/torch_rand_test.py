import unittest

import numpy as np
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.rand",
    ref="https://pytorch.org/docs/stable/generated/torch.rand.html",
)
class RandTest(unittest.TestCase):
    def test_scalar(self) -> None:
        torch_eggs.assert_tensor_structure(
            torch.rand([]),
            0.0,
        )

        torch_eggs.assert_tensor_structure(
            torch.rand(2, 3),
            torch.zeros(2, 3),
        )

    def test_stats(self) -> None:
        k = 1000

        with torch_eggs.reset_generator_seed():
            samples = torch.rand(k)

        eggs.assert_close_to(
            samples.mean(),
            0.5,
            rtol=0.05,
        )
        eggs.assert_close_to(
            samples.min(),
            0.0,
            delta=0.01,
        )
        eggs.assert_close_to(
            samples.max(),
            1.0,
            rtol=0.05,
        )


@api_link(
    target="torch.rand_like",
    ref="https://pytorch.org/docs/stable/generated/torch.rand_like.html",
)
class RandLikeTest(unittest.TestCase):
    def test_scalar(self) -> None:
        torch_eggs.assert_tensor_structure(
            torch.rand_like(torch.tensor(0.0)),
            0.0,
        )

        torch_eggs.assert_tensor_structure(
            torch.rand_like(torch.zeros(2, 3)),
            torch.zeros(2, 3),
        )

    def test_stats(self) -> None:
        k = 1000

        with torch_eggs.reset_generator_seed():
            samples = torch.rand_like(torch.zeros(k))

        eggs.assert_close_to(
            samples.mean(),
            0.5,
            rtol=0.05,
        )
        eggs.assert_close_to(
            samples.min(),
            0.0,
            delta=0.01,
        )
        eggs.assert_close_to(
            samples.max(),
            1.0,
            rtol=0.05,
        )


@api_link(
    target="torch.randint",
    ref="https://pytorch.org/docs/stable/generated/torch.randint.html",
)
class RandintTest(unittest.TestCase):
    def test_scalar(self) -> None:
        torch_eggs.assert_tensor_structure(
            torch.randint(1, size=[]),
            0,
        )

        torch_eggs.assert_tensor_structure(
            torch.randint(1, size=[2, 3]),
            torch.zeros(2, 3, dtype=torch.int64),
        )

    def test_stats(self) -> None:
        k = 1000
        # inclusive
        min = 3
        max = 10

        with torch_eggs.reset_generator_seed():
            samples = torch.randint(low=min, high=max + 1, size=[k])

        eggs.assert_close_to(
            torch.as_tensor(samples, dtype=torch.float).mean(),
            np.mean(range(min, max + 1)),
            rtol=0.05,
        )
        eggs.assert_close_to(
            samples.min(),
            min,
            delta=0.01,
        )
        eggs.assert_close_to(
            samples.max(),
            max,
            rtol=0.05,
        )


@api_link(
    target="torch.randint_like",
    ref="https://pytorch.org/docs/stable/generated/torch.randint_like.html",
)
class RandintLikeTest(unittest.TestCase):
    def test_scalar(self) -> None:
        torch_eggs.assert_tensor_structure(
            torch.randint_like(
                torch.ones(size=[]),
                1,
            ),
            0.0,
        )

        torch_eggs.assert_tensor_structure(
            torch.randint_like(
                torch.ones(size=[2, 3]),
                1,
            ),
            torch.zeros(2, 3),
        )

    def test_stats(self) -> None:
        k = 1000
        # inclusive
        min = 3
        max = 10

        with torch_eggs.reset_generator_seed():
            samples = torch.randint_like(
                torch.ones(k),
                low=min,
                high=max + 1,
            )

        eggs.assert_close_to(
            torch.as_tensor(samples, dtype=torch.float).mean(),
            np.mean(range(min, max + 1)),
            rtol=0.05,
        )
        eggs.assert_close_to(
            samples.min(),
            min,
            delta=0.01,
        )
        eggs.assert_close_to(
            samples.max(),
            max,
            rtol=0.05,
        )


@api_link(
    target="torch.randn",
    ref="https://pytorch.org/docs/stable/generated/torch.randn.html",
)
class RandnTest(unittest.TestCase):
    def test_scalar(self) -> None:
        torch_eggs.assert_tensor_structure(
            torch.randn([]),
            0.0,
        )

        torch_eggs.assert_tensor_structure(
            torch.randn(2, 3),
            torch.zeros(2, 3),
        )

    def test_stats(self) -> None:
        k = 10000

        with torch_eggs.reset_generator_seed():
            samples = torch.randn(k)

        eggs.assert_close_to(
            samples.mean(),
            0.0,
            atol=0.05,
        )
        eggs.assert_close_to(
            samples.var(),
            1.0,
            rtol=0.05,
        )


@api_link(
    target="torch.randn_like",
    ref="https://pytorch.org/docs/stable/generated/torch.randn_like.html",
)
class RandnLikeTest(unittest.TestCase):
    def test_scalar(self) -> None:
        torch_eggs.assert_tensor_structure(
            torch.randn_like(torch.tensor(0.0)),
            0.0,
        )

        torch_eggs.assert_tensor_structure(
            torch.randn_like(torch.zeros(2, 3)),
            torch.zeros(2, 3),
        )

    def test_stats(self) -> None:
        k = 10000

        with torch_eggs.reset_generator_seed():
            samples = torch.randn_like(torch.zeros(k))

        eggs.assert_close_to(
            samples.mean(),
            0.0,
            atol=0.05,
        )
        eggs.assert_close_to(
            samples.var(),
            1.0,
            rtol=0.05,
        )
