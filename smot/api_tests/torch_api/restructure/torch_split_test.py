import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class SplitTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.split.html"
    TARGET = torch.split

    def test_split(self) -> None:
        source = torch.arange(10).reshape(5, 2)

        torch_eggs.assert_view_tensor_seq(
            torch.split(source, 2),
            source,
            [
                [0, 1],
                [2, 3],
            ],
            [
                [4, 5],
                [6, 7],
            ],
            [[8, 9]],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.split(source, 1, dim=1),
            source,
            [[0], [2], [4], [6], [8]],
            [[1], [3], [5], [7], [9]],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.split(source, [1, 4]),
            source,
            [[0, 1]],
            [
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
            ],
        )
