import unittest

import pytest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.full",
    ref="https://pytorch.org/docs/stable/generated/torch.full.html",
)
class FullTest(unittest.TestCase):
    def test_full_scalar(self) -> None:
        torch_eggs.assert_tensor_equals(
            torch.full(tuple(), 2),
            torch.tensor(2),
        )

    def test_full(self) -> None:
        for dtype in [torch.int8, torch.float32]:
            torch_eggs.assert_tensor_equals(
                torch.full((3,), 2, dtype=dtype),
                torch.tensor([2, 2, 2], dtype=dtype),
            )

    @pytest.mark.slow
    def test_full_cuda(self) -> None:
        if torch.cuda.is_available():
            for dtype in [torch.int8, torch.float32]:
                torch_eggs.assert_tensor_equals(
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
