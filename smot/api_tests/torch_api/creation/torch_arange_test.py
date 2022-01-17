import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class ArangeTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.arange.html"
    TARGET = torch.arange

    def test_arange(self) -> None:
        torch_eggs.assert_tensor(
            torch.arange(5),
            [0, 1, 2, 3, 4],
        )

    def test_arange_int_dtype(self) -> None:
        for dtype in [torch.int8, torch.float32]:
            torch_eggs.assert_tensor(
                torch.arange(5, dtype=dtype),
                torch.tensor([0, 1, 2, 3, 4], dtype=dtype),
            )

            torch_eggs.assert_tensor(
                torch.arange(1, 4, dtype=dtype),
                torch.tensor([1, 2, 3], dtype=dtype),
            )

    def test_arange_float(self) -> None:
        torch_eggs.assert_tensor(
            torch.arange(1, 2.5, 0.5),
            [1, 1.5, 2.0],
        )

        torch_eggs.assert_tensor(
            torch.arange(1, 2.5 + 0.1, 0.5),
            [1, 1.5, 2.0, 2.5],
        )
