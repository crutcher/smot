import unittest

import hamcrest
import numpy as np
import torch

from smot.testing import eggs
from smot.torch.testing import torch_eggs


class TensorTest(unittest.TestCase):
    def test_scalar_tensor(self):
        t = torch.tensor(1)

        eggs.assert_match(
            t.item(),
            1,
        )

        eggs.assert_match(
            t.shape,
            torch.Size([]),
        )

        eggs.assert_match(
            t.numel(),
            1,
        )

    def test_copy(self):
        source = [1, 2]
        t = torch.tensor(source)
        torch_eggs.assert_tensor(t, source)
        eggs.assert_match(
            t.data,
            hamcrest.not_(hamcrest.same_instance(source)),
        )

        source = torch.tensor([1, 2])
        # t = torch.tensor(source)
        t = source.clone().detach()
        torch_eggs.assert_tensor(t, source)
        eggs.assert_match(
            t.data,
            hamcrest.not_(hamcrest.same_instance(source)),
        )

        source = np.array([1, 2])
        torch_eggs.assert_tensor(t, source)
        eggs.assert_match(
            t.data,
            hamcrest.not_(hamcrest.same_instance(source)),
        )

    def disable_test_create_pinned(self):
        # this is expensive.
        t = torch.tensor([1], pin_memory=True)
        eggs.assert_truthy(t.is_pinned())
