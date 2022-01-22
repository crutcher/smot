import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.split",
    ref="https://pytorch.org/docs/stable/generated/torch.split.html",
)
class SplitTest(unittest.TestCase):
    def test_split(self) -> None:
        source = torch.arange(10).reshape(5, 2)

        torch_eggs.assert_view_tensor_seq(
            torch.split(source, 2),
            source,
            [
                [0, 1],
                [2, 3],
            ],
            [
                [4, 5],
                [6, 7],
            ],
            [[8, 9]],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.split(source, 1, dim=1),
            source,
            [[0], [2], [4], [6], [8]],
            [[1], [3], [5], [7], [9]],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.split(source, [1, 4]),
            source,
            [[0, 1]],
            [
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
            ],
        )
