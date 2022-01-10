import unittest

import torch

from smot.pytorch_tree.testlib import torch_eggs


class FrombufferTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.frombuffer.html

    def test_frombuffer(self):
        source = bytearray([0, 1, 2, 3, 4])

        view = torch.frombuffer(
            source,
            count=3,
            offset=1,
            dtype=torch.int8,
        )

        torch_eggs.assert_tensor(
            view,
            torch.tensor([1, 2, 3], dtype=torch.int8),
        )

        # mutations to one mutate the other.
        source[1] = 8

        torch_eggs.assert_tensor(
            view,
            torch.tensor([8, 2, 3], dtype=torch.int8),
        )

        # anything that re-allocates the buffer is not safe:
        source.extend([20, 21, 22, 23] * 250)
        try:
            torch_eggs.assert_tensor(
                view,
                torch.tensor([2, 3, 4], dtype=torch.int8),
            )
        except AssertionError:
            # this is expected to fail.
            pass
