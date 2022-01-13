import unittest

import torch

from smot.testlib import torch_eggs


class LogspaceTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.logspace.html

    def test_logspace(self):
        torch_eggs.assert_tensor(
            torch.logspace(-10, 10, steps=5),
            [1.0e-10, 1.0e-05, 1.0e00, 1.0e05, 1.0e10],
        )

        # negative steps.
        torch_eggs.assert_tensor(
            torch.logspace(10, -10, steps=5),
            [1.0e10, 1.0e05, 1.0e00, 1.0e-05, 1.0e-10],
        )

    def test_logspace_100(self):
        # steps=100 as a default is deprecated.
        torch_eggs.assert_tensor(
            torch.logspace(1, 100),
            [float(f"1.e{i}") for i in range(1, 101)],
        )
