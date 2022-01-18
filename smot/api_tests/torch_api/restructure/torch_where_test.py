import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class WhereTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.where.html"
    TARGET = torch.where

    def test_where(self) -> None:
        a = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        b = torch.tensor(
            [
                [10, -1, -2],
                [-3, 11, 12],
            ]
        )

        torch_eggs.assert_tensor(
            torch.where(a > b, a, b),
            [
                [10, 2, 3],
                [4, 11, 12],
            ],
        )

    def test_scalar(self) -> None:
        a = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )

        torch_eggs.assert_tensor(
            torch.where(a < 4, a, 10),
            [
                [1, 2, 3],
                [10, 10, 10],
            ],
        )

    def test_broadcasting(self) -> None:
        a = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        b = torch.tensor(
            [
                [10, -1, -2],
            ]
        )

        torch_eggs.assert_tensor(
            torch.where(a > b, a, b),
            [
                [10, 2, 3],
                [10, 5, 6],
            ],
        )
