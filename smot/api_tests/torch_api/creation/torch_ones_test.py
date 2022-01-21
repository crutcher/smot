import unittest

import hamcrest
import torch

from smot.api_tests.doc_links import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.ones",
    ref="https://pytorch.org/docs/stable/generated/torch.ones.html",
)
class OnesTest(unittest.TestCase):
    def test_default(self) -> None:
        t = torch.ones(1, 2)

        torch_eggs.assert_tensor(
            t,
            [[1.0, 1.0]],
        )

    def test_scalar(self) -> None:
        # torch.ones(size) doesn't have a default;
        # but you can still construct a scalar.
        t = torch.ones(size=[])

        eggs.assert_match(t.size(), torch.Size([]))
        eggs.assert_match(t.numel(), 1)
        eggs.assert_match(t.item(), 1)

    def test_out(self) -> None:
        out = torch.tensor([[3.0, 4.0]])
        original_data = out.data_ptr()

        # verify that the original has the data we think.
        torch_eggs.assert_tensor(
            out,
            [[3.0, 4.0]],
        )

        # using out=<tensor> writes the ones to the target tensor.

        # Not changing the size writes in-place:
        torch.ones(1, 2, out=out)

        torch_eggs.assert_tensor(
            out,
            [[1.0, 1.0]],
        )

        eggs.assert_match(
            out.data_ptr(),
            original_data,
        )

        # NOTE: changing the size allocates a new data Tensor!
        torch.ones(1, 3, out=out)

        torch_eggs.assert_tensor(
            out,
            [[1.0, 1.0, 1.0]],
        )

        eggs.assert_match(
            out.data_ptr(),
            hamcrest.not_(original_data),
        )
