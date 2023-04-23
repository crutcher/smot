import unittest

import torch

from smot.api_tests.torch_api.math.torch_eggs_op_testlib import (
    assert_cellwise_bin_op_returns,
    assert_cellwise_unary_op_returns,
)
from smot.doc_link.link_annotations import WEIRD_API, api_link


class BitwiseOpsTest(unittest.TestCase):
    @api_link(
        target="torch.bitwise_and",
        ref="https://pytorch.org/docs/stable/generated/torch.bitwise_and.html",
    )
    @api_link(
        target="torch.Tensor.bitwise_and",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_and.html",
    )
    def test_bitwise_and(self) -> None:
        for op, supports_out in [
            (torch.bitwise_and, True),
            (torch.Tensor.bitwise_and, False),
        ]:
            for input, other, expected in [
                (
                    torch.tensor([False, False, True, True], dtype=torch.bool),
                    torch.tensor([False, True, False, True], dtype=torch.bool),
                    torch.tensor([False, False, False, True], dtype=torch.bool),
                ),
                (
                    torch.tensor([0x1, 0x1 | 0x2, 0x1 | 0x2 | 0x8], dtype=torch.int32),
                    torch.tensor([0x1, 0x2, 0x2 | 0x4 | 0x8], dtype=torch.int32),
                    torch.tensor([0x1, 0x2, 0x2 | 0x8], dtype=torch.int32),
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op, input, other, expected=expected, supports_out=supports_out
                )

    @api_link(
        target="torch.bitwise_not",
        ref="https://pytorch.org/docs/stable/generated/torch.bitwise_not.html",
    )
    @api_link(
        target="torch.Tensor.bitwise_not",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_not.html",
    )
    def test_bitwise_not(self) -> None:
        for op, supports_out in [
            (torch.bitwise_not, True),
            (torch.Tensor.bitwise_not, False),
        ]:
            for input, expected in [
                (
                    torch.tensor(True, dtype=torch.bool),
                    torch.tensor(False, dtype=torch.bool),
                ),
                (
                    torch.tensor([False, True], dtype=torch.bool),
                    torch.tensor([True, False], dtype=torch.bool),
                ),
                (
                    torch.tensor(0xFA, dtype=torch.uint8),
                    torch.tensor(0x05, dtype=torch.uint8),
                ),
                (
                    torch.tensor(0x0AFAFAFA, dtype=torch.int32),
                    torch.tensor(-0xAFAFAFB, dtype=torch.int32),
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected=expected,
                    supports_out=supports_out,
                )

    @api_link(
        target="torch.bitwise_or",
        ref="https://pytorch.org/docs/stable/generated/torch.bitwise_or.html",
    )
    @api_link(
        target="torch.Tensor.bitwise_or",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_or.html",
    )
    def test_bitwise_or(self) -> None:
        for op, supports_out in [
            (torch.bitwise_or, True),
            (torch.Tensor.bitwise_or, False),
        ]:
            for input, other, expected in [
                (
                    torch.tensor([False, False, True, True], dtype=torch.bool),
                    torch.tensor([False, True, False, True], dtype=torch.bool),
                    torch.tensor([False, True, True, True], dtype=torch.bool),
                ),
                (
                    torch.tensor([0x1, 0x1 | 0x2, 0x1 | 0x2 | 0x8], dtype=torch.int32),
                    torch.tensor([0x1, 0x2, 0x2 | 0x4 | 0x8], dtype=torch.int32),
                    torch.tensor(
                        [0x1, 0x1 | 0x2, 0x1 | 0x2 | 0x4 | 0x8], dtype=torch.int32
                    ),
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op, input, other, expected=expected, supports_out=supports_out
                )

    @api_link(
        target="torch.bitwise_xor",
        ref="https://pytorch.org/docs/stable/generated/torch.bitwise_xor.html",
    )
    @api_link(
        target="torch.Tensor.bitwise_xor",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_xor.html",
    )
    def test_bitwise_xor(self) -> None:
        for op, supports_out in [
            (torch.bitwise_xor, True),
            (torch.Tensor.bitwise_xor, False),
        ]:
            for input, other, expected in [
                (
                    torch.tensor([False, False, True, True], dtype=torch.bool),
                    torch.tensor([False, True, False, True], dtype=torch.bool),
                    torch.tensor([False, True, True, False], dtype=torch.bool),
                ),
                (
                    torch.tensor([0x1, 0x1 | 0x2, 0x1 | 0x2 | 0x8], dtype=torch.int32),
                    torch.tensor([0x1, 0x2, 0x2 | 0x4 | 0x8], dtype=torch.int32),
                    torch.tensor([0x0, 0x1, 0x1 | 0x4], dtype=torch.int32),
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op, input, other, expected=expected, supports_out=supports_out
                )

    @api_link(
        target="torch.bitwise_left_shift",
        ref="https://pytorch.org/docs/stable/generated/torch.bitwise_left_shift.html",
    )
    @api_link(
        target="torch.Tensor.bitwise_left_shift",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_left_shift.html",
    )
    def test_bitwise_left_shift(self) -> None:
        for op, supports_out in [
            (torch.bitwise_left_shift, True),
            (torch.Tensor.bitwise_left_shift, False),
        ]:
            WEIRD_API(
                target="torch.bitwise_left_shift",
                note="conversion shift dtype affects overflow / output type.",
            )
            for input, other, expected in [
                (
                    torch.tensor(0x1, dtype=torch.uint8),
                    torch.tensor(7, dtype=torch.uint8),
                    torch.tensor(0x80, dtype=torch.uint8),
                ),
                (
                    torch.tensor(0x1, dtype=torch.uint8),
                    torch.tensor(8, dtype=torch.uint8),
                    # conversion shift dtype affects overflow / output type:
                    torch.tensor(0, dtype=torch.uint8),
                ),
                (
                    torch.tensor(0x1, dtype=torch.uint8),
                    torch.tensor(8, dtype=torch.int8),
                    # conversion shift dtype affects overflow / output type:
                    torch.tensor(0x100, dtype=torch.int16),
                ),
                (
                    torch.tensor([0x1, 0x1, 0x2, 0x3], dtype=torch.int32),
                    torch.tensor([0, 1, 0, 3], dtype=torch.int32),
                    torch.tensor([0x1, 0x2, 0x2, 0x18], dtype=torch.int32),
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op, input, other, expected=expected, supports_out=supports_out
                )

    @api_link(
        target="torch.bitwise_right_shift",
        ref="https://pytorch.org/docs/stable/generated/torch.bitwise_right_shift.html",
    )
    @api_link(
        target="torch.Tensor.bitwise_right_shift",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.bitwise_right_shift.html",
    )
    def test_bitwise_right_shift(self) -> None:
        for op, supports_out in [
            (torch.bitwise_right_shift, True),
            (torch.Tensor.bitwise_right_shift, False),
        ]:
            for input, other, expected in [
                (
                    torch.tensor(0x80, dtype=torch.uint8),
                    torch.tensor(7, dtype=torch.uint8),
                    torch.tensor(0x1, dtype=torch.uint8),
                ),
                (
                    # Unlike left_shift, underflow is not determined by shift dtype.
                    # See left_shift weirdness.
                    torch.tensor(0x1, dtype=torch.uint8),
                    torch.tensor(1, dtype=torch.uint8),
                    torch.tensor(0, dtype=torch.uint8),
                ),
                (
                    # Unlike left_shift, underflow is not determined by shift dtype.
                    # See left_shift weirdness.
                    torch.tensor(0x1, dtype=torch.uint8),
                    torch.tensor(1, dtype=torch.int8),
                    torch.tensor(0, dtype=torch.int16),
                ),
                (
                    torch.tensor([0x1, 0x2, 0x2, 0x18], dtype=torch.int32),
                    torch.tensor([0, 1, 0, 3], dtype=torch.int32),
                    torch.tensor([0x1, 0x1, 0x2, 0x3], dtype=torch.int32),
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op, input, other, expected=expected, supports_out=supports_out
                )
