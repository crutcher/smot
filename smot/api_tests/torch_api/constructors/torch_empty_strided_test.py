import unittest

import torch

from smot.testlib import eggs, torch_eggs


class EmptyStridedTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.empty_strided.html

    def test_empty_strided_scalar(self):
        eggs.assert_raises(
            lambda: torch.empty_strided(1, 1),  # type: ignore
            TypeError,
            "must be tuple of ints, not int",
        )

    def test_empty_strided(self):
        torch_eggs.assert_tensor_structure(
            torch.empty_strided((2, 3), (1, 2)),
            torch.ones(2, 3),
        )

        # out of order strides ...
        torch_eggs.assert_tensor_structure(
            torch.empty_strided((2, 3), (2, 1)),
            torch.ones(2, 3),
        )

        torch_eggs.assert_tensor_structure(
            torch.empty_strided((2, 3), (1, 4)),
            torch.ones(2, 3),
        )
