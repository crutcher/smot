import unittest

import torch

from smot.testlib import torch_eggs


class PermuteTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.permute.html

    def test_permute(self):
        source = torch.tensor(
            [
                [
                    [1, 2],
                    [3, 4],
                ],
                [
                    [5, 6],
                    [7, 8],
                ],
            ]
        )

        view = torch.permute(source, (2, 0, 1))

        torch_eggs.assert_view(view, source)

        torch_eggs.assert_tensor(
            view,
            [
                [
                    [1, 3],
                    [5, 7],
                ],
                [
                    [2, 4],
                    [6, 8],
                ],
            ],
        )
