import unittest

import pytest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.zeros_like",
    ref="https://pytorch.org/docs/stable/generated/torch.zeros_like.html",
)
class ZerosLikeTest(unittest.TestCase):
    def impl(self, device: str) -> None:
        # Dense Tensors
        for dtype in [torch.int8, torch.float32]:
            source = torch.tensor(
                [1, 2],
                dtype=dtype,
                device=device,
            )

            torch_eggs.assert_tensor_equals(
                torch.zeros_like(source),
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
                dtype=dtype,
            )

            torch_eggs.assert_tensor_equals(
                torch.zeros_like(source),
                torch.sparse_coo_tensor(
                    indices=torch.zeros(size=(2, 0)),
                    values=torch.zeros(size=(0,)),
                    size=source.size(),
                    dtype=source.dtype,
                    device=source.device,
                ),
            )

    def test_cpu(self) -> None:
        self.impl("cpu")

    @pytest.mark.slow
    def test_cuda(self) -> None:
        if torch.cuda.is_available():
            self.impl("cuda")
