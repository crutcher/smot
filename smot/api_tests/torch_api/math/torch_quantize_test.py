import unittest

import torch

from smot.api_tests.torch_api.math.torch_eggs_op_testlib import (
    assert_cellwise_op_returns,
)
from smot.doc_link.link_annotations import api_link


class TorchQuantizeTest(unittest.TestCase):
    @api_link(
        target="torch.fake_quantize_per_channel_affine",
        ref="https://pytorch.org/docs/stable/generated/torch.fake_quantize_per_tensor_affine.html",
    )
    def test_entr(self) -> None:
        input = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        scale = torch.tile(torch.tensor(0.5), (5,))
        zero_point = torch.tile(torch.tensor(0, dtype=torch.int32), (5,))

        expected = [0.0, 0.0, 0.5, 1.0, 1.0]

        assert_cellwise_op_returns(
            torch.fake_quantize_per_channel_affine,
            input,
            scale,
            zero_point,
            axis=0,
            quant_min=0,
            quant_max=255,
            expected=expected,
            close=True,
            # does not support 'out='
            supports_out=False,
        )
