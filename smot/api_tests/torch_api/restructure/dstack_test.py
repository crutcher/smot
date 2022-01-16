import unittest

import torch

from smot.testlib import eggs, torch_eggs


class DstackTest(unittest.TestCase):
    def test_dstack(self):
        torch_eggs.assert_tensor(
            torch.dstack(
                (
                    torch.tensor([1, 2, 3]),
                    torch.tensor([4, 5, 6]),
                )
            ),
            [
                [
                    [1, 4],
                    [2, 5],
                    [3, 6],
                ]
            ],
        )

        torch_eggs.assert_tensor(
            torch.dstack(
                (
                    torch.tensor([[1], [2], [3]]),
                    torch.tensor([[4], [5], [6]]),
                )
            ),
            [
                [[1, 4]],
                [[2, 5]],
                [[3, 6]],
            ],
        )

        torch_eggs.assert_tensor(
            torch.dstack(
                (
                    torch.tensor([[[1, 2]], [[3, 4]]]),
                    torch.tensor([[5], [6]]),
                )
            ),
            [
                [[1, 2, 5]],
                [[3, 4, 6]],
            ],
        )

    def test_out(self):
        target = torch.arange(6)
        orig_data_ptr = target.data_ptr()

        torch.dstack(
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
            ),
            out=target,
        )

        torch_eggs.assert_tensor(
            target,
            [
                [
                    [1, 4],
                    [2, 5],
                    [3, 6],
                ]
            ],
        )

        eggs.assert_match(target.data_ptr(), orig_data_ptr)
