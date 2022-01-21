import unittest

import torch

from smot.api_tests.doc_links import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.transpose",
    ref="https://pytorch.org/docs/stable/generated/torch.transpose.html",
)
class TransposeTest(unittest.TestCase):
    def test_transpose(self) -> None:
        source = torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
        )
        torch_eggs.assert_view_tensor(
            torch.transpose(source, 0, 1),
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
            lambda: torch.transpose(source, 0, 3),
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

        torch_eggs.assert_view_tensor(
            torch.transpose(source, 0, 2),
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
