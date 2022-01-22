import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.permute",
    ref="https://pytorch.org/docs/stable/generated/torch.permute.html",
)
class PermuteTest(unittest.TestCase):
    def test_permute(self) -> None:
        source = torch.tensor(
            [
                [
                    [1, 2],
                    [3, 4],
                ],
                [
                    [5, 6],
                    [7, 8],
                ],
            ]
        )

        view = torch.permute(source, (2, 0, 1))

        torch_eggs.assert_views(source, view)

        torch_eggs.assert_tensor(
            view,
            [
                [
                    [1, 3],
                    [5, 7],
                ],
                [
                    [2, 4],
                    [6, 8],
                ],
            ],
        )
