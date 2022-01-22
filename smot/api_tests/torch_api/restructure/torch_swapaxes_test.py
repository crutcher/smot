import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.swapaxes",
    ref="https://pytorch.org/docs/stable/generated/torch.swapaxes.html",
    alias=["torch.transpose", "torch.swapaxes"],
)
class SwapaxesTest(unittest.TestCase):
    def test_swapaxes(self) -> None:
        source = torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
        )
        torch_eggs.assert_view_tensor(
            torch.swapaxes(source, 0, 1),
            source,
            [
                [0, 3],
                [1, 4],
                [2, 5],
            ],
        )

    def test_error(self) -> None:
        source = torch.ones(2, 3)
        eggs.assert_raises(
            lambda: torch.swapaxes(source, 0, 3),
            IndexError,
            "Dimension out of range",
        )

    def test_swapaxes_3d(self) -> None:
        source = torch.tensor(
            [
                [
                    [[0, 1], [2, 3], [4, 5]],
                    [[6, 7], [8, 9], [10, 11]],
                ]
            ],
        )

        torch_eggs.assert_view_tensor(
            torch.swapaxes(source, 0, 2),
            source,
            [
                [
                    [[0, 1]],
                    [[6, 7]],
                ],
                [
                    [[2, 3]],
                    [[8, 9]],
                ],
                [
                    [[4, 5]],
                    [[10, 11]],
                ],
            ],
        )
