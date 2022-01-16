import unittest

import torch

from smot.testlib import torch_eggs


class HsplitTest(unittest.TestCase):
    def test_view(self):
        source = torch.arange(16.0).reshape(4, 4)
        torch_eggs.assert_tensor(
            source,
            torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ]
            ),
        )

        # when input.ndims > 1;
        # hsplit(input, indices) => tensor_split(input, indices, dim=1)
        splits = torch.hsplit(source, 2)
        torch_eggs.assert_tensor_seq(
            splits,
            *torch.tensor_split(source, 2, dim=1),
        )
        torch_eggs.assert_tensor_seq(
            splits,
            [
                [0.0, 1.0],
                [4.0, 5.0],
                [8.0, 9.0],
                [12.0, 13.0],
            ],
            [
                [2.0, 3.0],
                [6.0, 7.0],
                [10.0, 11.0],
                [14.0, 15.0],
            ],
        )

        # split is a view:
        source[0, 0] = 77.0
        splits[0][0, 1] = 88.0

        torch_eggs.assert_tensor_seq(
            splits,
            [
                [77.0, 88.0],
                [4.0, 5.0],
                [8.0, 9.0],
                [12.0, 13.0],
            ],
            [
                [2.0, 3.0],
                [6.0, 7.0],
                [10.0, 11.0],
                [14.0, 15.0],
            ],
        )

        torch_eggs.assert_tensor(
            source,
            torch.tensor(
                [
                    [77.0, 88.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ]
            ),
        )

    def test_indices(self):
        source = torch.arange(16.0).reshape(4, 4)
        torch_eggs.assert_tensor(
            source,
            torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ]
            ),
        )

        splits = torch.hsplit(source, [1, 3])
        torch_eggs.assert_tensor_seq(
            splits,
            *torch.tensor_split(source, [1, 3], dim=1),
        )
        torch_eggs.assert_tensor_seq(
            splits,
            [
                [0.0],
                [4.0],
                [8.0],
                [12.0],
            ],
            [
                [1.0, 2.0],
                [5.0, 6.0],
                [9.0, 10.0],
                [13.0, 14.0],
            ],
            [
                [3.0],
                [7.0],
                [11.0],
                [15.0],
            ],
        )
