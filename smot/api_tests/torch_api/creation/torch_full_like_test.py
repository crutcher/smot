import unittest

import pytest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.full_like",
    ref="https://pytorch.org/docs/stable/generated/torch.full_like.html",
)
class FullLikeTest(unittest.TestCase):
    def test_full_like_scalar(self) -> None:
        t = torch.tensor(0)

        torch_eggs.assert_tensor_structure(
            torch.full_like(t, 2),
            torch.tensor(2),
        )

    def test_full_like(self) -> None:
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
    def test_full_like_cuda(self) -> None:
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
