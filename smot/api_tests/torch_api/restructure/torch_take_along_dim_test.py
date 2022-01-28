import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.take_along_dim",
    ref="https://pytorch.org/docs/stable/generated/torch.take_along_dim.html",
)
class TakeAlongDimTest(unittest.TestCase):
    def test_take(self) -> None:
        src = torch.tensor([[10, 30, 20], [60, 40, 50]])

        max_idx = torch.argmax(src)
        torch_eggs.assert_tensor_equals(
            max_idx,
            # no dim, acts like take
            3,
        )
        torch_eggs.assert_tensor_equals(
            torch.take_along_dim(src, max_idx),
            [60],
        )

        sorted_idx = torch.argsort(src, dim=1)
        torch_eggs.assert_tensor_equals(
            sorted_idx,
            [[0, 2, 1], [1, 2, 0]],
        )
        torch_eggs.assert_tensor_equals(
            torch.take_along_dim(src, sorted_idx, dim=1),
            [[10, 20, 30], [40, 50, 60]],
        )

    def test_error(self) -> None:
        src = torch.tensor([[10, 30, 20], [60, 40, 50]])

        eggs.assert_raises(
            lambda: torch.take_along_dim(src, torch.tensor([3]), dim=2),
            RuntimeError,
            "input and indices should have the same number of dimensions",
        )
