import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.tile",
    ref="https://pytorch.org/docs/stable/generated/torch.tile.html",
)
class TileTest(unittest.TestCase):
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
