import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.randperm",
    ref="https://pytorch.org/docs/stable/generated/torch.randperm.html",
)
class RandpermTest(unittest.TestCase):
    def test_degenerate(self) -> None:
        torch_eggs.assert_tensor_structure(
            torch.randperm(0, dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        )

    def test_one(self) -> None:
        torch_eggs.assert_tensor_structure(
            torch.randperm(1, dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )

    def test_n(self) -> None:
        n = 5
        perm = torch.randperm(n)
        for k in range(n):
            eggs.assert_match(
                torch.count_nonzero(perm == k),
                1,
            )
