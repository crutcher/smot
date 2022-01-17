import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class RowStackTest(TorchApiTestCase):
    "torch.row_stack is an alias for torch.vstack"

    API_DOC = "https://pytorch.org/docs/stable/generated/torch.row_stack.html"
    TARGET = torch.row_stack
    ALIAS_FOR = torch.vstack

    def test_row_stack(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])

        torch_eggs.assert_tensor(
            torch.row_stack((a, b)),
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
        )

        a = torch.tensor([[1], [2], [3]])
        b = torch.tensor([[4], [5], [6]])

        torch_eggs.assert_tensor(
            torch.row_stack((a, b)),
            [
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
            ],
        )
