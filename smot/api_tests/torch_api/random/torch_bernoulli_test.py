import unittest

import hamcrest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.bernoulli",
    ref="https://pytorch.org/docs/stable/generated/torch.bernoulli.html",
)
class BernoulliTest(unittest.TestCase):
    def test_basic(self) -> None:
        g = torch.Generator()
        g.manual_seed(12345)

        # coin flip, one is always 1, 0 is always 0.
        # shape is determined by input.

        torch_eggs.assert_tensor_equals(
            torch.bernoulli(torch.ones(3, 3, dtype=torch.float32), generator=g),
            torch.ones(3, 3),
        )

        torch_eggs.assert_tensor_equals(
            torch.bernoulli(torch.zeros(3, 3, dtype=torch.float32), generator=g),
            torch.zeros(3, 3),
        )

        torch_eggs.assert_tensor_equals(
            torch.bernoulli(torch.tensor([0.0, 1.0, 0.0, 1.0]), generator=g),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
        )

        k = 1000
        for p in [0.2, 0.5, 0.83]:
            eggs.assert_match(
                torch.sum(torch.bernoulli(torch.full((k,), p), generator=g)),
                hamcrest.close_to(k * p, k * 0.05),
            )

    @api_link(
        target="torch.Tensor.bernoulli_",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.bernoulli_.html",
        note="`t.bernoulli_(...)` => `torch.bernoulli(..., out=t)`",
    )
    def test_out(self) -> None:
        g = torch.Generator()
        g.manual_seed(12345)

        k = 1000
        t = torch.empty((k,))
        for p in [0.2, 0.5, 0.83]:
            # in place
            torch.bernoulli(torch.full((k,), p), out=t, generator=g)

            eggs.assert_match(
                torch.sum(t),
                hamcrest.close_to(k * p, k * 0.05),
            )

            torch.zeros((k,), out=t)

            # t.bernoulli_(...) => torch.bernoulli(..., out=t)
            t.bernoulli_(p, generator=g)

            eggs.assert_match(
                torch.sum(t),
                hamcrest.close_to(k * p, k * 0.05),
            )
