import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class VstackTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.vstack.html"
    TARGET = torch.vstack

    def test_vstack(self) -> None:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])

        torch_eggs.assert_tensor(
            torch.vstack((a, b)),
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
        )

        a = torch.tensor([[1], [2], [3]])
        b = torch.tensor([[4], [5], [6]])

        torch_eggs.assert_tensor(
            torch.vstack((a, b)),
            [
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
            ],
        )
