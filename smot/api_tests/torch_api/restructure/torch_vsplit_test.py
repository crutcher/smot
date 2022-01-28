import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.vsplit",
    ref="https://pytorch.org/docs/stable/generated/torch.vsplit.html",
)
class VsplitTest(unittest.TestCase):
    def test_vsplit(self) -> None:
        source = torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ]
        )

        torch_eggs.assert_tensor_sequence_equals(
            torch.vsplit(source, 2),
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
            torch.vsplit(source, 4),
            [
                [0, 1, 2, 3],
            ],
            [
                [4, 5, 6, 7],
            ],
            [
                [8, 9, 10, 11],
            ],
            [
                [12, 13, 14, 15],
            ],
            view_of=source,
        )

        torch_eggs.assert_tensor_sequence_equals(
            torch.vsplit(source, (1, 3)),
            [
                [0, 1, 2, 3],
            ],
            [
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            [
                [12, 13, 14, 15],
            ],
            view_of=source,
        )

    def test_error(self) -> None:
        source = torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ]
        )

        eggs.assert_raises(
            lambda: torch.vsplit(source, 3),
            RuntimeError,
            "the size of the dimension 4 is not divisible by",
        )
