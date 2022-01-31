import unittest

import torch

from smot.api_tests.torch_api.math.torch_eggs_op_testlib import (
    assert_cellwise_bin_op_returns,
    assert_cellwise_unary_op_returns,
    assert_tensor_uniop_not_implemented,
)
from smot.doc_link.link_annotations import api_link


class MathOpTest(unittest.TestCase):
    @api_link(
        target="torch.abs",
        ref="https://pytorch.org/docs/stable/generated/torch.abs.html",
    )
    @api_link(
        target="torch.absolute",
        ref="https://pytorch.org/docs/stable/generated/torch.absolute.html",
        alias="torch.abs",
    )
    @api_link(
        target="torch.Tensor.abs",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.abs.html",
    )
    @api_link(
        target="torch.Tensor.absolute",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.absolute.html",
        alias="torch.Tensor.abs",
    )
    def test_abs(self) -> None:
        for op, supports_out in [
            (torch.abs, True),
            (torch.absolute, True),
            (torch.Tensor.abs, False),
            (torch.Tensor.absolute, False),
        ]:
            for input, expected in [
                (
                    torch.tensor(1, dtype=torch.int64),
                    torch.tensor(1, dtype=torch.int64),
                ),
                (
                    torch.tensor(-1, dtype=torch.int64),
                    torch.tensor(1, dtype=torch.int64),
                ),
                (
                    torch.tensor(1.0, dtype=torch.float64),
                    torch.tensor(1.0, dtype=torch.float64),
                ),
                (
                    torch.tensor(-1.0, dtype=torch.float32),
                    torch.tensor(1.0, dtype=torch.float32),
                ),
                (
                    torch.tensor(torch.nan, dtype=torch.float64),
                    torch.tensor(torch.nan, dtype=torch.float64),
                ),
                (
                    torch.tensor(-3 + 4j, dtype=torch.complex128),
                    torch.tensor(5.0, dtype=torch.float64),
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected,
                    supports_out=supports_out,
                )

            for not_implemented in [
                [True, False],
            ]:
                assert_tensor_uniop_not_implemented(op, not_implemented)

    @api_link(
        target="torch.acos",
        ref="https://pytorch.org/docs/stable/generated/torch.acos.html",
    )
    @api_link(
        target="torch.arccos",
        ref="https://pytorch.org/docs/stable/generated/torch.arccos.html",
        alias="torch.acos",
    )
    @api_link(
        target="torch.Tensor.acos",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.acos.html",
    )
    @api_link(
        target="torch.Tensor.arccos",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.arccos.html",
        alias="torch.Tensor.acos",
    )
    def test_acos(self) -> None:
        for op, supports_out in [
            (torch.acos, True),
            (torch.arccos, True),
            (torch.Tensor.acos, False),
            (torch.Tensor.arccos, False),
        ]:
            for input, expected in [
                (
                    torch.tensor(1, dtype=torch.int64),
                    torch.tensor(0.0, dtype=torch.float32),
                ),
                (
                    torch.tensor(-1, dtype=torch.int64),
                    torch.tensor(torch.pi, dtype=torch.float32),
                ),
                (
                    torch.tensor(1.0, dtype=torch.float64),
                    torch.tensor(0.0, dtype=torch.float64),
                ),
                (
                    torch.tensor(torch.nan, dtype=torch.float64),
                    torch.tensor(torch.nan, dtype=torch.float64),
                ),
                (
                    # Boolean values are cast to float { 0.0, 1.0 } inputs.
                    [False, True],
                    [1.5707963705, 0.0],
                ),
                (
                    [0j, 1 + 1j],
                    [
                        1.5707963705 - 0.0000000000j,
                        0.9045568705 - 1.0612751245j,
                    ],
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected,
                    close=True,
                    supports_out=supports_out,
                )

    @api_link(
        target="torch.asin",
        ref="https://pytorch.org/docs/stable/generated/torch.asin.html",
    )
    @api_link(
        target="torch.arcsin",
        ref="https://pytorch.org/docs/stable/generated/torch.arcsin.html",
        alias="torch.asin",
    )
    @api_link(
        target="torch.Tensor.asin",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.asin.html",
    )
    @api_link(
        target="torch.Tensor.arcsin",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.arcsin.html",
        alias="torch.Tensor.asin",
    )
    def test_asin(self) -> None:
        for op, supports_out in [
            (torch.asin, True),
            (torch.arcsin, True),
            (torch.Tensor.asin, False),
            (torch.Tensor.arcsin, False),
        ]:
            for input, expected in [
                (
                    # int64 => float32
                    torch.tensor(1, dtype=torch.int64),
                    torch.tensor(1.5707963705, dtype=torch.float32),
                ),
                (
                    # int64 => float32
                    torch.tensor(-1, dtype=torch.int64),
                    torch.tensor(-1.5707963705, dtype=torch.float32),
                ),
                (
                    torch.tensor(1.0, dtype=torch.float64),
                    torch.tensor(1.5707963705, dtype=torch.float64),
                ),
                (
                    torch.tensor(torch.nan, dtype=torch.float64),
                    torch.tensor(torch.nan, dtype=torch.float64),
                ),
                (
                    # Boolean values => float32
                    [False, True],
                    torch.tensor([0.0, 1.5707963705], dtype=torch.float32),
                ),
                (
                    [0j, 1 + 1j],
                    [0.0j, 0.6662394404 + 1.0612752438j],
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected,
                    close=True,
                    supports_out=supports_out,
                )

    @api_link(
        target="torch.asinh",
        ref="https://pytorch.org/docs/stable/generated/torch.asinh.html",
    )
    @api_link(
        target="torch.arcsinh",
        ref="https://pytorch.org/docs/stable/generated/torch.arcsinh.html",
        alias="torch.asinh",
    )
    @api_link(
        target="torch.Tensor.asinh",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.asinh.html",
    )
    @api_link(
        target="torch.Tensor.arcsinh",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.arcsinh.html",
        alias="torch.Tensor.asinh",
    )
    def test_asinh(self) -> None:
        for op, supports_out in [
            (torch.asinh, True),
            (torch.arcsinh, True),
            (torch.Tensor.asinh, False),
            (torch.Tensor.arcsinh, False),
        ]:
            for input, expected in [
                (
                    # int64 => float32
                    torch.tensor([0, 1, -1, 3], dtype=torch.int64),
                    [
                        0.0000000000,
                        0.8813735843,
                        -0.8813735843,
                        1.8184465170,
                    ],
                ),
                (
                    # int64 => float32
                    torch.tensor(-1, dtype=torch.int64),
                    torch.tensor(-0.8813735843, dtype=torch.float32),
                ),
                (
                    torch.tensor(1.0, dtype=torch.float64),
                    torch.tensor(0.8813735843, dtype=torch.float64),
                ),
                (
                    torch.tensor(torch.nan, dtype=torch.float64),
                    torch.tensor(torch.nan, dtype=torch.float64),
                ),
                (
                    # Boolean values => float32
                    [False, True],
                    torch.tensor([0.0000000000, 0.8813735843], dtype=torch.float32),
                ),
                (
                    [0j, 1 + 1j],
                    [
                        0.0000000000 + 0.0000000000j,
                        1.0612751245 + 0.6662394404j,
                    ],
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected,
                    close=True,
                    supports_out=supports_out,
                )

    @api_link(
        target="torch.acosh",
        ref="https://pytorch.org/docs/stable/generated/torch.acosh.html",
    )
    @api_link(
        target="torch.arccosh",
        ref="https://pytorch.org/docs/stable/generated/torch.arccosh.html",
        alias="torch.acosh",
    )
    @api_link(
        target="torch.Tensor.acosh",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.acosh.html",
    )
    @api_link(
        target="torch.Tensor.arccosh",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.arccosh.html",
        alias="torch.Tensor.acosh",
    )
    def test_acosh(self) -> None:
        for op, supports_out in [
            (torch.acosh, True),
            (torch.arccosh, True),
            (torch.Tensor.acosh, False),
            (torch.Tensor.arccosh, False),
        ]:
            for input, expected in [
                (
                    # int64 => float32
                    torch.tensor(
                        [
                            0,
                            1,
                            -1,
                            3,
                        ],
                        dtype=torch.int64,
                    ),
                    torch.tensor(
                        [
                            torch.nan,
                            0.0,
                            torch.nan,
                            1.7627471685,
                        ],
                        dtype=torch.float32,
                    ),
                ),
                (
                    # int64 => float32
                    [
                        torch.pi,
                        2 * torch.pi,
                    ],
                    [
                        1.8115262985,
                        2.5246307850,
                    ],
                ),
                (
                    # Boolean values => float32
                    [False, True],
                    torch.tensor([torch.nan, 0.0000000000], dtype=torch.float32),
                ),
                (
                    [0j, 1 + 1j],
                    [
                        0.0000000000 + 1.5707963705j,
                        1.0612751245 + 0.9045568705j,
                    ],
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected,
                    close=True,
                    supports_out=supports_out,
                )

    @api_link(
        target="torch.atan",
        ref="https://pytorch.org/docs/stable/generated/torch.atan.html",
    )
    @api_link(
        target="torch.arctan",
        ref="https://pytorch.org/docs/stable/generated/torch.arctan.html",
        alias="torch.atan",
    )
    @api_link(
        target="torch.Tensor.atan",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.atan.html",
    )
    @api_link(
        target="torch.Tensor.arctan",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.arctan.html",
        alias="torch.Tensor.atan",
    )
    def test_atan(self) -> None:
        for op, supports_out in [
            (torch.atan, True),
            (torch.arctan, True),
            (torch.Tensor.atan, False),
            (torch.Tensor.arctan, False),
        ]:
            for input, expected in [
                (
                    [0, 1, torch.pi, 2],
                    [0.0000000000, 0.7853981853, 1.2626272440, 1.1071487665],
                ),
                (
                    [True, False],
                    [0.7853981853, 0.0000000000],
                ),
                (
                    [0j, 1 + 1j],
                    [
                        0.0000000000 + 0.0000000000j,
                        1.0172219276 + 0.4023594856j,
                    ],
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected,
                    close=True,
                    supports_out=supports_out,
                )

    @api_link(
        target="torch.atanh",
        ref="https://pytorch.org/docs/stable/generated/torch.atanh.html",
    )
    @api_link(
        target="torch.arctanh",
        ref="https://pytorch.org/docs/stable/generated/torch.arctanh.html",
        alias="torch.atanh",
    )
    @api_link(
        target="torch.Tensor.atanh",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.atanh.html",
    )
    @api_link(
        target="torch.Tensor.arctanh",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.arctanh.html",
        alias="torch.Tensor.atanh",
    )
    def test_atanh(self) -> None:
        for op, supports_out in [
            (torch.atanh, True),
            (torch.arctanh, True),
            (torch.Tensor.atanh, False),
            (torch.Tensor.arctanh, False),
        ]:
            for input, expected in [
                (
                    [0, 1, torch.pi, 0.2],
                    [0.0000000000, torch.inf, torch.nan, 0.2027325481],
                ),
                (
                    [False, True],
                    [0.0, torch.inf],
                ),
                (
                    [0j, 1 + 1j],
                    [
                        0.0000000000 + 0.0000000000j,
                        0.4023594856 + 1.0172219276j,
                    ],
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected,
                    close=True,
                    supports_out=supports_out,
                )

    @api_link(
        target="torch.atan2",
        ref="https://pytorch.org/docs/stable/generated/torch.atan2.html",
    )
    @api_link(
        target="torch.Tensor.atan2",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.atan2.html",
    )
    def test_atan2(self) -> None:
        for op, supports_out in [
            (torch.atan2, True),
            (torch.Tensor.atan2, False),
        ]:
            for input, other, expected in [
                (
                    [0.0, torch.pi, torch.pi],
                    [0.0, 1.0, torch.pi],
                    # yields.
                    [0.0000000000, 1.2626272440, 0.7853981853],
                ),
                (
                    [False, True, True],
                    [0.0, 1.0, torch.pi],
                    # yields.
                    [0.0000000000, 0.7853981853, 0.3081690669],
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op,
                    input,
                    other,
                    expected,
                    close=True,
                    supports_out=supports_out,
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
                    torch.tensor(0xFA, dtype=torch.int8),
                    torch.tensor(0x05, dtype=torch.int8),
                ),
                (
                    torch.tensor(0xFAFAFAFA, dtype=torch.int32),
                    torch.tensor(0x05050505, dtype=torch.int32),
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected,
                    supports_out=supports_out,
                )

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
                    op,
                    input,
                    other,
                    expected,
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
                    op,
                    input,
                    other,
                    expected,
                    supports_out=supports_out,
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
                    op,
                    input,
                    other,
                    expected,
                    supports_out=supports_out,
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
                    op,
                    input,
                    other,
                    expected,
                    supports_out=supports_out,
                )
