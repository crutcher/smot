import unittest

import pytest
import torch

from smot.testlib import torch_eggs


class FullTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.full.html

    def test_full_scalar(self):
        torch_eggs.assert_tensor(
            torch.full(tuple(), 2),
            torch.tensor(2),
        )

    def test_full(self):
        for dtype in [torch.int8, torch.float32]:
            torch_eggs.assert_tensor(
                torch.full((3,), 2, dtype=dtype),
                torch.tensor([2, 2, 2], dtype=dtype),
            )

    @pytest.mark.slow
    def test_full_cuda(self):
        if torch.cuda.is_available():
            for dtype in [torch.int8, torch.float32]:
                torch_eggs.assert_tensor(
                    torch.full(
                        (3,),
                        2,
                        dtype=dtype,
                        device="cuda",
                    ),
                    torch.tensor(
                        [2, 2, 2],
                        dtype=dtype,
                        device="cuda",
                    ),
                )


class FullLikeTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.full_like.html

    def test_full_like_scalar(self):
        t = torch.tensor(0)

        torch_eggs.assert_tensor_structure(
            torch.full_like(t, 2),
            torch.tensor(2),
        )

    def test_full_like(self):
        for dtype in [torch.int8, torch.float32]:
            t = torch.tensor([1], dtype=dtype)
            torch_eggs.assert_tensor(
                torch.full_like(t, 2),
                torch.tensor([2], dtype=dtype),
            )

            t = torch.tensor([[1], [1]], dtype=dtype)
            torch_eggs.assert_tensor(
                torch.full_like(t, 2),
                torch.tensor([[2], [2]], dtype=dtype),
            )

    @pytest.mark.slow
    def test_full_like_cuda(self):
        if torch.cuda.is_available():
            for dtype in [torch.int8, torch.float32]:
                t = torch.tensor([1], dtype=dtype, device="cuda")
                torch_eggs.assert_tensor(
                    torch.full_like(t, 2),
                    torch.tensor([2], dtype=dtype, device="cuda"),
                )

                t = torch.tensor([[1], [1]], dtype=dtype, device="cuda")
                torch_eggs.assert_tensor(
                    torch.full_like(t, 2),
                    torch.tensor([[2], [2]], dtype=dtype, device="cuda"),
                )
