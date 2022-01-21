import unittest

import torch

from smot.api_tests.doc_links import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.vstack",
    ref="https://pytorch.org/docs/stable/generated/torch.vstack.html",
)
class VstackTest(unittest.TestCase):
    def test_vstack(self) -> None:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])

        torch_eggs.assert_tensor(
            torch.vstack((a, b)),
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
        )

        a = torch.tensor([[1], [2], [3]])
        b = torch.tensor([[4], [5], [6]])

        torch_eggs.assert_tensor(
            torch.vstack((a, b)),
            [
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
            ],
        )
