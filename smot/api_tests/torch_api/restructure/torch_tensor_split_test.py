import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class TensorSplitTest(TorchApiTestCase):
    API_DOCS = "https://pytorch.org/docs/stable/generated/torch.tensor_split.html"
    TARGET = torch.tensor_split

    def test_tensor_split(self) -> None:
        source = torch.tensor(
            [
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                ],
                [
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
            ]
        )

        torch_eggs.assert_view_tensor_seq(
            torch.tensor_split(source, 2),
            source,
            [
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                ],
            ],
            [
                [
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
            ],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.tensor_split(source, 2, dim=2),
            source,
            [
                [
                    [0, 1],
                    [4, 5],
                ],
                [
                    [8, 9],
                    [12, 13],
                ],
            ],
            [
                [
                    [2, 3],
                    [6, 7],
                ],
                [
                    [10, 11],
                    [14, 15],
                ],
            ],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.tensor_split(source, (1, 3), dim=2),
            source,
            [
                [
                    [0],
                    [4],
                ],
                [
                    [8],
                    [12],
                ],
            ],
            [
                [
                    [1, 2],
                    [5, 6],
                ],
                [
                    [9, 10],
                    [13, 14],
                ],
            ],
            [
                [
                    [3],
                    [7],
                ],
                [
                    [11],
                    [15],
                ],
            ],
        )
