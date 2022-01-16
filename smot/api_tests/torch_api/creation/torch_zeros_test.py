import unittest

import hamcrest
import torch

from smot.testlib import eggs, torch_eggs


class ZerosTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.zeros.html

    def test_default(self):
        t = torch.zeros(1, 2)

        torch_eggs.assert_tensor(
            t,
            [[0.0, 0.0]],
        )

    def test_scalar(self):
        # torch.zeros(size) doesn't have a default;
        # but you can still construct a scalar.
        t = torch.zeros(size=[])

        eggs.assert_match(t.size(), torch.Size([]))
        eggs.assert_match(t.numel(), 1)
        eggs.assert_match(t.item(), 0)

    def test_out(self):
        out = torch.tensor([[3.0, 4.0]])
        original_data = out.data_ptr()

        # verify that the original has the data we think.
        torch_eggs.assert_tensor(
            out,
            [[3.0, 4.0]],
        )

        # using out=<tensor> writes the zeros to the target tensor.

        # Not changing the size writes in-place:
        torch.zeros(1, 2, out=out)

        torch_eggs.assert_tensor(
            out,
            [[0.0, 0.0]],
        )

        eggs.assert_match(
            out.data_ptr(),
            original_data,
        )

        # NOTE: changing the size allocates a new data Tensor!
        torch.zeros(1, 3, out=out)

        torch_eggs.assert_tensor(
            out,
            [[0.0, 0.0, 0.0]],
        )

        eggs.assert_match(
            out.data_ptr(),
            hamcrest.not_(original_data),
        )
