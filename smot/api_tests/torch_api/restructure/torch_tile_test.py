import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class TileTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.tile.html"
    TARGET = torch.tile

    def test_tile(self) -> None:
        source = torch.tensor([[1, 2], [3, 4]])

        torch_eggs.assert_tensor(
            torch.tile(source, (2,)),
            [
                [1, 2, 1, 2],
                [3, 4, 3, 4],
            ],
        )

        torch_eggs.assert_tensor(
            torch.tile(source, (1, 2)),
            [
                [1, 2, 1, 2],
                [3, 4, 3, 4],
            ],
        )

        torch_eggs.assert_tensor(
            torch.tile(source, (2, 1)),
            [
                [1, 2],
                [3, 4],
                [1, 2],
                [3, 4],
            ],
        )

        torch_eggs.assert_tensor(
            torch.tile(source, (2, 1, 1)),
            [
                [
                    [1, 2],
                    [3, 4],
                ],
                [
                    [1, 2],
                    [3, 4],
                ],
            ],
        )

        torch_eggs.assert_tensor(
            torch.tile(source, (2, 2, 3)),
            [
                [
                    [1, 2, 1, 2, 1, 2],
                    [3, 4, 3, 4, 3, 4],
                    [1, 2, 1, 2, 1, 2],
                    [3, 4, 3, 4, 3, 4],
                ],
                [
                    [1, 2, 1, 2, 1, 2],
                    [3, 4, 3, 4, 3, 4],
                    [1, 2, 1, 2, 1, 2],
                    [3, 4, 3, 4, 3, 4],
                ],
            ],
        )
