import unittest

import torch

from smot.testlib import eggs, torch_eggs


class MoveaxisTest(unittest.TestCase):
    # alias for movedim
    # https://pytorch.org/docs/stable/generated/torch.moveaxis.html

    def test_moveaxis_int(self):
        source = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ]
        )

        eggs.assert_match(
            source.size(),
            torch.Size([3, 2, 2]),
        )

        view = torch.moveaxis(source, 1, 0)

        torch_eggs.assert_view(view, source)

        torch_eggs.assert_tensor(
            view,
            [
                [[1, 2], [5, 6], [9, 10]],
                [[3, 4], [7, 8], [11, 12]],
            ],
        )

        eggs.assert_match(
            view.size(),
            torch.Size([2, 3, 2]),
        )

    def test_moveaxis_tuples(self):
        source = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ]
        )

        eggs.assert_match(
            source.size(),
            torch.Size([3, 2, 2]),
        )

        view = torch.moveaxis(source, (1, 2), (0, 1))

        torch_eggs.assert_view(view, source)

        torch_eggs.assert_tensor(
            view,
            [
                [[1, 5, 9], [2, 6, 10]],
                [[3, 7, 11], [4, 8, 12]],
            ],
        )

        eggs.assert_match(
            view.size(),
            torch.Size([2, 2, 3]),
        )
