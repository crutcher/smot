import pytest
import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class EmptyLikeTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.empty_like.html"
    TARGET = torch.empty_like

    def test_empty_like_scalar(self):
        t = torch.tensor(0)

        torch_eggs.assert_tensor_structure(
            torch.empty_like(t),
            torch.tensor(0),
        )

    def test_empty_like(self):
        for dtype in [torch.int8, torch.float32]:
            for data in [0, [[0]], [[1], [2]]]:
                t = torch.tensor(data, dtype=dtype)

                torch_eggs.assert_tensor_structure(
                    torch.empty_like(t),
                    t,
                )

    @pytest.mark.slow
    def test_empty_like_cuda(self):
        if torch.cuda.is_available():
            for dtype in [torch.int8, torch.float32]:
                for data in [0, [[0]], [[1], [2]]]:
                    t = torch.tensor(data, dtype=dtype, device="cuda")

                    torch_eggs.assert_tensor_structure(
                        torch.empty_like(t),
                        t,
                    )
