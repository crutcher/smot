import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.take",
    ref="https://pytorch.org/docs/stable/generated/torch.take.html",
)
class TakeTest(unittest.TestCase):
    def test_take(self) -> None:
        src = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )

        torch_eggs.assert_tensor_equals(
            torch.take(src, torch.tensor([0, 2, 3])),
            [1, 3, 4],
        )

        torch_eggs.assert_tensor_equals(
            torch.take(src, torch.tensor([[0, 2], [3, 5]])),
            [[1, 3], [4, 6]],
        )

        view = src.t()
        torch_eggs.assert_tensor_equals(
            view, expected=[[1, 4], [2, 5], [3, 6]], view_of=src
        )

        # indexing is done 1-d in the view's indexing.

        torch_eggs.assert_tensor_equals(
            torch.take(view, torch.tensor([0, 2, 3])),
            [1, 2, 5],
        )

        torch_eggs.assert_tensor_equals(
            torch.take(view, torch.tensor([[0, 2], [3, 5]])),
            [[1, 2], [5, 6]],
        )
