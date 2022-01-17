import unittest

import torch

from smot.testlib import torch_eggs


class RowStackTest(unittest.TestCase):
    # torch.row_stack is an alias for torch.vstack
    # https://pytorch.org/docs/stable/generated/torch.row_stack.html

    def test_row_stack(self):
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
