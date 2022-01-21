import unittest

import torch

from smot.api_tests.doc_links import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.quantize_per_tensor",
    ref="https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html",
)
class QuantizePerTensorTest(unittest.TestCase):
    def test_quantize_per_tensor(self) -> None:
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

    def test_quantize_per_tensor_list(self) -> None:
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
