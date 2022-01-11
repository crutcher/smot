import unittest

import hamcrest
import pytest
import torch

from smot.pytorch_tree.testlib import torch_eggs
from smot.testlib import eggs


class ZerosTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.zeros.html

    def test_default(self):
        t = torch.zeros(1, 2)

        torch_eggs.assert_tensor(
            t,
            torch.tensor([[0.0, 0.0]]),
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
            torch.tensor([[3.0, 4.0]]),
        )

        # using out=<tensor> writes the zeros to the target tensor.

        # Not changing the size writes in-place:
        torch.zeros(1, 2, out=out)

        torch_eggs.assert_tensor(
            out,
            torch.tensor([[0.0, 0.0]]),
        )

        eggs.assert_match(
            out.data_ptr(),
            original_data,
        )

        # NOTE: changing the size allocates a new data Tensor!
        torch.zeros(1, 3, out=out)

        torch_eggs.assert_tensor(
            out,
            torch.tensor([[0.0, 0.0, 0.0]]),
        )

        eggs.assert_match(
            out.data_ptr(),
            hamcrest.not_(original_data),
        )


class ZerosLikeTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.zeros_like.html

    def impl(self, device):
        # Dense Tensors
        for dtype in [torch.int8, torch.float32]:
            source = torch.tensor(
                [1, 2],
                dtype=dtype,
                device=device,
            )
            z = torch.zeros_like(source)

            torch_eggs.assert_tensor(
                z,
                torch.zeros(
                    *source.size(),
                    dtype=source.dtype,
                    device=source.device,
                    layout=source.layout,
                ),
            )

        # Sparse COO Tensors
        for dtype in [torch.int8, torch.float32]:
            coo = torch.tensor([[0, 0], [2, 2]])
            vals = torch.tensor([3, 4])
            source = torch.sparse_coo_tensor(
                indices=coo,
                values=vals,
                device=device,
            )

            z = torch.zeros_like(source)
            c = z.coalesce()

            expected = torch.sparse_coo_tensor(
                indices=torch.zeros(size=(2, 0)),
                values=torch.zeros(size=(0,)),
                size=source.size(),
                dtype=source.dtype,
                device=source.device,
            ).coalesce()

            torch_eggs.assert_tensor(c, expected)

    def test_cpu(self):
        self.impl("cpu")

    @pytest.mark.slow
    def test_cuda(self):
        if torch.cuda.is_available():
            self.impl("cuda")
