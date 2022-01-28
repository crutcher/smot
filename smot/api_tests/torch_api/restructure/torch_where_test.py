import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.where",
    ref="https://pytorch.org/docs/stable/generated/torch.where.html",
)
class WhereTest(unittest.TestCase):
    def test_where(self) -> None:
        a = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        b = torch.tensor(
            [
                [10, -1, -2],
                [-3, 11, 12],
            ]
        )

        torch_eggs.assert_tensor_equals(
            torch.where(a > b, a, b),
            [
                [10, 2, 3],
                [4, 11, 12],
            ],
        )

    def test_scalar(self) -> None:
        a = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )

        torch_eggs.assert_tensor_equals(
            torch.where(a < 4, a, 10),
            [
                [1, 2, 3],
                [10, 10, 10],
            ],
        )

    def test_broadcasting(self) -> None:
        a = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        b = torch.tensor(
            [
                [10, -1, -2],
            ]
        )

        torch_eggs.assert_tensor_equals(
            torch.where(a > b, a, b),
            [
                [10, 2, 3],
                [10, 5, 6],
            ],
        )
