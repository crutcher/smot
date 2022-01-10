import unittest

import torch

from smot.pytorch_tree.testlib import torch_eggs
from smot.testlib import eggs


class ZerosTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.zeros.html

    def test_default(self):
        t = torch.zeros(1, 2)

        torch_eggs.assert_tensor(
            t,
            torch.tensor([[0.0, 0.0]]),
        )

    def test_scalar(self):
        # torch.zeros(size) doesn't have a default;
        # but you can still construct a scalar.
        t = torch.zeros(size=[])

        eggs.assert_match(t.size(), torch.Size([]))
        eggs.assert_match(t.numel(), 1)
        eggs.assert_match(t.item(), 0)
