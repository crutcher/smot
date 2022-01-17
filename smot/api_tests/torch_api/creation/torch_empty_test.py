import pytest
import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class EmptyTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.empty.html"
    TARGET = torch.empty

    def test_empty_zero(self):
        torch_eggs.assert_tensor(
            torch.empty(0),
            torch.ones(0),
        )

    def test_empty(self):
        for dtype in [torch.int8, torch.float32]:
            torch_eggs.assert_tensor_structure(
                torch.empty(3, dtype=dtype),
                torch.tensor(
                    # random data ...
                    [0, 0, 0],
                    dtype=dtype,
                ),
            )

    @pytest.mark.slow
    def test_empty_cuda(self):
        if torch.cuda.is_available():
            for dtype in [torch.int8, torch.float32]:
                torch_eggs.assert_tensor_structure(
                    torch.empty(
                        3,
                        dtype=dtype,
                        device="cuda",
                    ),
                    torch.tensor(
                        # random data ...
                        [0, 0, 0],
                        dtype=dtype,
                        device="cuda",
                    ),
                )
