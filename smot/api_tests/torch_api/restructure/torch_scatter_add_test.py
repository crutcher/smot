import unittest

import torch

from smot.testlib import torch_eggs


class ScatterTest(unittest.TestCase):
    # out-of-place variant of torch.Tensor.scatter_add_
    # https://pytorch.org/docs/stable/generated/torch.scatter_add.html

    # Note: the backward pass is implemented only for src.shape == index.shape

    def test_add(self):
        source = torch.arange(9, dtype=torch.int64).reshape(3, 3)

        torch_eggs.assert_tensor(
            torch.scatter_add(
                source,
                1,
                torch.tensor(
                    [
                        [2],
                        [1],
                        [0],
                    ],
                ),
                source,
            ),
            [
                [0, 1, 2],
                [3, 7, 5],
                [12, 7, 8],
            ],
        )
