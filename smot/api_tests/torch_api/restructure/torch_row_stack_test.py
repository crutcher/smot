import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.row_stack",
    ref="https://pytorch.org/docs/stable/generated/torch.row_stack.html",
    note="torch.row_stack is an alias for torch.vstack",
)
class RowStackTest(unittest.TestCase):
    def test_row_stack(self) -> None:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])

        torch_eggs.assert_tensor(
            torch.row_stack((a, b)),
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
        )

        a = torch.tensor([[1], [2], [3]])
        b = torch.tensor([[4], [5], [6]])

        torch_eggs.assert_tensor(
            torch.row_stack((a, b)),
            [
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
            ],
        )
