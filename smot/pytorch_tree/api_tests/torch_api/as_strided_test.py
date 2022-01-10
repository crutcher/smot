import unittest

import hamcrest
import torch

from smot.pytorch_tree.testlib import torch_eggs
from smot.testlib import eggs


class AsStridedTest(unittest.TestCase):
    def test_as_strided(self):
        t = torch.rand(3, 3)
        v = t.data.storage()

        x = torch.as_strided(t, (2, 2), (1, 2))

        eggs.assert_match(
            x.data_ptr(),
            t.data_ptr(),
        )

        torch_eggs.assert_tensor(
            x,
            [
                [v[1 * 0 + 2 * 0], v[1 * 0 + 2 * 1]],
                [v[1 * 1 + 2 * 0], v[1 * 1 + 2 * 1]],
            ],
        )

    def test_storage_offset(self):
        t = torch.rand(3, 3, dtype=torch.float32)
        v = t.data.storage()
        offset = 2

        x = torch.as_strided(t, (2, 2), (1, 2), 2)

        eggs.assert_match(
            x.data_ptr(),
            t.data_ptr() + 4 * offset,
        )

        torch_eggs.assert_tensor(
            x,
            [
                [v[2 + 1 * 0 + 2 * 0], v[2 + 1 * 0 + 2 * 1]],
                [v[2 + 1 * 1 + 2 * 0], v[2 + 1 * 1 + 2 * 1]],
            ],
        )
