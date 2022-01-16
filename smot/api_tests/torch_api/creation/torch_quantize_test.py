import unittest

import torch

from smot.testlib import torch_eggs


class QuantizePerTensorTest(unittest.TestCase):
    def test_quantize_per_tensor(self):
        source = torch.tensor([-1.0, 0.0, 1.0, 2.0])

        t = torch.quantize_per_tensor(
            input=source,
            scale=0.1,
            zero_point=10,
            dtype=torch.qint8,
        )

        # TODO: how to read the actual quantization state?

        torch_eggs.assert_tensor(
            t.int_repr(),
            torch.tensor([0, 10, 20, 30], dtype=torch.int8),
        )

        torch_eggs.assert_tensor(
            torch.dequantize(t),
            source,
        )

    def test_quantize_per_tensor_list(self):
        # NOTE!
        # when used on multiple inputs, the keywords change:
        #  * input => tensors
        #  * scale => scales
        #  * zero_point => zero_points
        ts = torch.quantize_per_tensor(
            tensors=[
                torch.tensor([-1.0, 0.0, 1.0, 2.0]),
                torch.tensor([-1.0, 0.0]),
            ],
            scales=torch.tensor([0.1, 0.5]),
            zero_points=torch.tensor([10, 5]),
            dtype=torch.qint8,
        )

        # TODO: how to read the actual quantization state?

        torch_eggs.assert_tensor(
            ts[0].int_repr(),
            torch.tensor([0, 10, 20, 30], dtype=torch.int8),
        )
        torch_eggs.assert_tensor(
            ts[1].int_repr(),
            torch.tensor([3, 5], dtype=torch.int8),
        )


class QuantizePerChannelTest(unittest.TestCase):
    def test_quantize_per_channel(self):
        source = torch.tensor(
            [
                [-1.0, 0.0],
                [1.0, 2.0],
            ],
        )
        t = torch.quantize_per_channel(
            input=source,
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

        torch_eggs.assert_tensor(
            torch.dequantize(t),
            source,
        )
