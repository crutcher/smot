import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class QuantizePerChannelTest(TorchApiTestCase):
    API_DOC = (
        "https://pytorch.org/docs/stable/generated/torch.quantize_per_channel.html"
    )
    TARGET = torch.quantize_per_channel

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
