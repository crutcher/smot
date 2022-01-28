import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.unbind",
    ref="https://pytorch.org/docs/stable/generated/torch.unbind.html",
)
class UnbindTest(unittest.TestCase):
    def test_unbind(self) -> None:
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

        # unbind returns views.

        torch_eggs.assert_tensor_sequence_equals(
            torch.unbind(source),
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ],
            [
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            view_of=source,
        )

        torch_eggs.assert_tensor_sequence_equals(
            torch.unbind(source, dim=1),
            [
                [0, 1, 2, 3],
                [8, 9, 10, 11],
            ],
            [
                [4, 5, 6, 7],
                [12, 13, 14, 15],
            ],
            view_of=source,
        )

        torch_eggs.assert_tensor_sequence_equals(
            torch.unbind(source, dim=2),
            [
                [0, 4],
                [8, 12],
            ],
            [
                [1, 5],
                [9, 13],
            ],
            [
                [2, 6],
                [10, 14],
            ],
            [
                [3, 7],
                [11, 15],
            ],
            view_of=source,
        )
