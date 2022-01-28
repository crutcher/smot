import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs
from smot.testlib.eggs import ignore_warnings


@api_link(
    target="torch.range",
    ref="https://pytorch.org/docs/stable/generated/torch.range.html",
)
class RangeTest(unittest.TestCase):
    def test_range(self) -> None:
        with ignore_warnings():
            torch_eggs.assert_tensor_equals(
                torch.range(0, 4),
                [0.0, 1.0, 2.0, 3.0, 4.0],
            )

    def test_range_int_dtype(self) -> None:
        with ignore_warnings():
            for dtype in [torch.int8, torch.float32]:
                torch_eggs.assert_tensor_equals(
                    torch.range(0, 4, dtype=dtype),
                    torch.tensor([0, 1, 2, 3, 4], dtype=dtype),
                )

    def test_range_float(self) -> None:
        with ignore_warnings():
            torch_eggs.assert_tensor_equals(
                torch.range(1, 2.5, 0.5),
                [1, 1.5, 2.0, 2.5],
            )

            torch_eggs.assert_tensor_equals(
                torch.range(1, 2.5 + 0.1, 0.5),
                [1, 1.5, 2.0, 2.5],
            )
