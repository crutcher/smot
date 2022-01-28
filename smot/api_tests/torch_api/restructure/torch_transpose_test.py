from typing import Callable, List
import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.transpose",
    ref="https://pytorch.org/docs/stable/generated/torch.transpose.html",
    alias=["torch.swapdims", "torch.swapaxes"],
)
@api_link(
    target="torch.swapaxes",
    ref="https://pytorch.org/docs/stable/generated/torch.swapaxes.html",
    alias=["torch.transpose", "torch.swapdims"],
)
@api_link(
    target="torch.swapdims",
    ref="https://pytorch.org/docs/stable/generated/torch.swapdims.html",
    alias=["torch.transpose", "torch.swapaxes"],
)
class TransposeTest(unittest.TestCase):
    ALIASES: List[Callable[[torch.Tensor, int, int], torch.Tensor]] = [
        torch.transpose,
        torch.swapaxes,
        torch.swapdims,
    ]

    def test_transpose(self) -> None:
        source = torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
        )
        for transpose in self.ALIASES:
            torch_eggs.assert_tensor_equals(
                transpose(source, 0, 1),
                expected=[
                    [0, 3],
                    [1, 4],
                    [2, 5],
                ],
                view_of=source,
            )

    def test_error(self) -> None:
        source = torch.ones(2, 3)
        for transpose in self.ALIASES:
            eggs.assert_raises(
                lambda: transpose(source, 0, 3),
                IndexError,
                "Dimension out of range",
            )

    def test_transpose_3d(self) -> None:
        source = torch.tensor(
            [
                [
                    [[0, 1], [2, 3], [4, 5]],
                    [[6, 7], [8, 9], [10, 11]],
                ]
            ],
        )

        for transpose in self.ALIASES:
            torch_eggs.assert_tensor_equals(
                transpose(source, 0, 2),
                expected=[
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
                view_of=source,
            )
