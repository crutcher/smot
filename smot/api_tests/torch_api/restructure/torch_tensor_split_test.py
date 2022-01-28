import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.tensor_split",
    ref="https://pytorch.org/docs/stable/generated/torch.tensor_split.html",
)
class TensorSplitTest(unittest.TestCase):
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

        torch_eggs.assert_tensor_sequence_equals(
            torch.tensor_split(source, 2),
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
            view_of=source,
        )

        torch_eggs.assert_tensor_sequence_equals(
            torch.tensor_split(source, 2, dim=2),
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
            view_of=source,
        )

        torch_eggs.assert_tensor_sequence_equals(
            torch.tensor_split(source, (1, 3), dim=2),
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
            view_of=source,
        )
