import unittest

import torch

from smot.testlib import torch_eggs


class QuantizePerChannelTest(unittest.TestCase):
    def test_quantize_per_channel(self):
        t = torch.quantize_per_channel(
            input=torch.tensor(
                [
                    [-1.0, 0.0],
                    [1.0, 2.0],
                ],
            ),
            scales=torch.tensor([0.1, 0.01]),
            zero_points=torch.tensor([10, 0]),
            axis=0,
            dtype=torch.quint8,
        )

        # TODO: how to read the actual quantization state?

        torch_eggs.assert_tensor(
            t.int_repr(),
            torch.tensor(
                [
                    [0, 10],
                    [100, 200],
                ],
                dtype=torch.uint8,
            ),
        )
