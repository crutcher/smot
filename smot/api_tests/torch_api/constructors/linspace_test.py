import unittest

import torch

from smot.testlib import torch_eggs


class LinspaceTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.linspace.html

    def test_linspace(self):
        torch_eggs.assert_tensor(
            torch.linspace(3, 10, steps=5),
            [3.0, 4.75, 6.5, 8.25, 10.0],
        )

        # negative steps.
        torch_eggs.assert_tensor(
            torch.linspace(10, 3, steps=5),
            [10.0, 8.25, 6.5, 4.75, 3.0],
        )

    def test_linspace_100(self):
        # steps=100 as a default is deprecated.
        torch_eggs.assert_tensor(
            torch.linspace(1, 100),
            [float(i) for i in range(1, 101)],
        )
