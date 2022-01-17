import unittest

import torch

from smot.testlib import torch_eggs


class ScatterTest(unittest.TestCase):
    # out-of-place variant of torch.Tensor.scatter_
    # https://pytorch.org/docs/stable/generated/torch.scatter.html

    # Note: the backward pass is implemented only for src.shape == index.shape

    def test_basic(self):
        a = torch.ones(3, 3, dtype=torch.int64)
        b = torch.arange(9, dtype=torch.int64).reshape(3, 3)

        idx = torch.tensor(
            [
                [0, 1, 2],
                [2, 1, 0],
                [1, 2, 0],
            ]
        )

        torch_eggs.assert_tensor(
            torch.scatter(a, 0, idx, b),
            [
                [0, 1, 8],
                [6, 4, 1],
                [3, 7, 2],
            ],
        )

    def test_reduce(self):
        b = torch.arange(9, dtype=torch.int64).reshape(3, 3)

        torch_eggs.assert_tensor(
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

        torch_eggs.assert_tensor(
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

        torch_eggs.assert_tensor(
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
