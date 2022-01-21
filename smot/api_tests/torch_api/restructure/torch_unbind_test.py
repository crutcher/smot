import unittest

import torch

from smot.api_tests.doc_links import api_link
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

        torch_eggs.assert_view_tensor_seq(
            torch.unbind(source),
            source,
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ],
            [
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.unbind(source, dim=1),
            source,
            [
                [0, 1, 2, 3],
                [8, 9, 10, 11],
            ],
            [
                [4, 5, 6, 7],
                [12, 13, 14, 15],
            ],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.unbind(source, dim=2),
            source,
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
        )
