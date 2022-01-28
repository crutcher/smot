import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.quantize_per_channel",
    ref="https://pytorch.org/docs/stable/generated/torch.quantize_per_channel.html",
)
class QuantizePerChannelTest(unittest.TestCase):
    def test_quantize_per_channel(self) -> None:
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

        torch_eggs.assert_tensor_equals(
            t.int_repr(),
            torch.tensor(
                [
                    [0, 10],
                    [100, 200],
                ],
                dtype=torch.uint8,
            ),
        )

        torch_eggs.assert_tensor_equals(
            torch.dequantize(t),
            source,
        )
