import pytest
import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class FullTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.full.html"
    TARGET = torch.full

    def test_full_scalar(self) -> None:
        torch_eggs.assert_tensor(
            torch.full(tuple(), 2),
            torch.tensor(2),
        )

    def test_full(self) -> None:
        for dtype in [torch.int8, torch.float32]:
            torch_eggs.assert_tensor(
                torch.full((3,), 2, dtype=dtype),
                torch.tensor([2, 2, 2], dtype=dtype),
            )

    @pytest.mark.slow
    def test_full_cuda(self) -> None:
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
