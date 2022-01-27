import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


class MathTest(unittest.TestCase):
    @api_link(
        target="torch.abs",
        ref="https://pytorch.org/docs/stable/generated/torch.abs.html",
    )
    def test_abs(self) -> None:
        torch_eggs.assert_tensor_uniop_pair_cases(
            torch.abs,
            torch.Tensor.abs,
            # ints
            (
                [[-1], [3]],
                [[1], [3]],
            ),
            # floats
            (
                [[-1.5], [3.5]],
                [[1.5], [3.5]],
            ),
            # complex
            # the complex abs is the l2 norm.
            (
                [[-1.5 + 0j], [-3 + 4j]],
                [[1.5], [5.0]],
            ),
            unsupported=[
                [True, False],
            ],
        )
