import unittest

import torch

from smot.testlib import eggs, torch_eggs


class ColumnStackTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.column_stack.html

    def test_column_stack(self):
        torch_eggs.assert_tensor(
            torch.column_stack(
                (
                    torch.tensor([1, 2, 3]),
                    torch.tensor([4, 5, 6]),
                ),
            ),
            [
                [1, 4],
                [2, 5],
                [3, 6],
            ],
        )

        torch_eggs.assert_tensor(
            torch.column_stack(
                (
                    torch.arange(5),
                    torch.arange(10).reshape(5, 2),
                    torch.arange(10).reshape(5, 2),
                ),
            ),
            [  # a, b, b, c, c
                [0, 0, 1, 0, 1],
                [1, 2, 3, 2, 3],
                [2, 4, 5, 4, 5],
                [3, 6, 7, 6, 7],
                [4, 8, 9, 8, 9],
            ],
        )

    def test_scalar_error(self):
        eggs.assert_raises(
            lambda: torch.column_stack(
                (
                    # The docs suggest you can use scalars,
                    # but you can't mix them
                    torch.tensor(77),
                    torch.tensor([1, 2, 3]),
                ),
            ),
            RuntimeError,
            "Sizes of tensors must match except in dimension 1",
        )

        # You can do this though:
        torch_eggs.assert_tensor(
            torch.column_stack(
                (
                    torch.tensor(3),
                    torch.tensor(4),
                    torch.tensor([[5, 6]]),
                ),
            ),
            [[3, 4, 5, 6]],
        )

    def test_out(self):
        target = torch.arange(6)
        orig_data_ptr = target.data_ptr()

        torch.column_stack(
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
            ),
            out=target,
        )

        torch_eggs.assert_tensor(
            target,
            [
                [1, 4],
                [2, 5],
                [3, 6],
            ],
        )

        eggs.assert_match(
            target.data_ptr(),
            orig_data_ptr,
        )
