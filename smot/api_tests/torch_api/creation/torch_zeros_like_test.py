import pytest
import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class ZerosLikeTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.zeros_like.html"
    TARGET = torch.zeros_like

    def impl(self, device: str) -> None:
        # Dense Tensors
        for dtype in [torch.int8, torch.float32]:
            source = torch.tensor(
                [1, 2],
                dtype=dtype,
                device=device,
            )

            torch_eggs.assert_tensor(
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

            torch_eggs.assert_tensor(
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
