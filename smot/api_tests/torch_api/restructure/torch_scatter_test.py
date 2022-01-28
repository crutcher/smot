import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.scatter",
    ref="https://pytorch.org/docs/stable/generated/torch.scatter.html",
    note="Note: the backward pass is implemented only for src.shape == index.shape",
)
class ScatterTest(unittest.TestCase):
    def test_basic(self) -> None:
        a = torch.ones(3, 3, dtype=torch.int64)
        b = torch.arange(9, dtype=torch.int64).reshape(3, 3)

        idx = torch.tensor(
            [
                [0, 1, 2],
                [2, 1, 0],
                [1, 2, 0],
            ]
        )

        torch_eggs.assert_tensor_equals(
            torch.scatter(a, 0, idx, b),
            [
                [0, 1, 8],
                [6, 4, 1],
                [3, 7, 2],
            ],
        )

    def test_reduce(self) -> None:
        b = torch.arange(9, dtype=torch.int64).reshape(3, 3)

        torch_eggs.assert_tensor_equals(
            torch.scatter(
                b,
                1,
                torch.tensor(
                    [
                        [2],
                        [1],
                        [0],
                    ],
                ),
                10,
                reduce="add",
            ),
            [
                [0, 1, 12],
                [3, 14, 5],
                [16, 7, 8],
            ],
        )

        torch_eggs.assert_tensor_equals(
            torch.scatter(
                b,
                1,
                torch.tensor(
                    [
                        [2],
                        [1],
                        [0],
                    ],
                ),
                b,
                reduce="add",
            ),
            [
                [0, 1, 2],
                [3, 7, 5],
                [12, 7, 8],
            ],
        )

        torch_eggs.assert_tensor_equals(
            torch.scatter(
                b,
                1,
                torch.tensor(
                    [
                        [0, 1, 2],
                        [0, 1, 2],
                        [0, 1, 2],
                    ],
                ),
                b,
                reduce="multiply",
            ),
            [
                [0, 1, 4],
                [9, 16, 25],
                [36, 49, 64],
            ],
        )
