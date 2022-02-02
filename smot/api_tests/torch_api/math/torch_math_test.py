import unittest

import torch

from smot.api_tests.torch_api.math.torch_eggs_op_testlib import (
    assert_cellwise_bin_op_returns,
    assert_cellwise_op_returns,
    assert_cellwise_unary_op_returns,
    assert_tensor_op_throws_not_implemented,
)
from smot.doc_link.link_annotations import WEIRD_API, api_link
from smot.testlib import torch_eggs


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
                    op, input, expected=expected, supports_out=supports_out
                )

            for not_implemented in [
                [True, False],
            ]:
                assert_tensor_op_throws_not_implemented(op, not_implemented)

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
                    op, input, expected=expected, supports_out=supports_out
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
                    op, input, other, expected=expected, supports_out=supports_out
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

    @api_link(
        target="torch.ceil",
        ref="https://pytorch.org/docs/stable/generated/torch.ceil.html",
    )
    @api_link(
        target="torch.Tensor.ceil",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.ceil.html",
    )
    def test_ceil(self) -> None:
        for op, supports_out in [
            (torch.ceil, True),
            (torch.Tensor.ceil, False),
        ]:
            for input, expected in [
                (
                    torch.tensor([-0.3, -0.0, 0.0, 0.3, 1.2, 4.0], dtype=torch.float64),
                    torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 4.0], dtype=torch.float64),
                ),
                (
                    torch.tensor([-0.3, -0.0, 0.0, 0.3, 1.2, 4.0], dtype=torch.float32),
                    torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 4.0], dtype=torch.float32),
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op, input, expected=expected, supports_out=supports_out
                )

            for not_implemented in [
                torch.tensor(0, dtype=torch.int64),
                torch.tensor(False, dtype=torch.bool),
                torch.tensor(0.4 - 0.3j, dtype=torch.complex128),
            ]:
                assert_tensor_op_throws_not_implemented(op, not_implemented)

    @api_link(
        target="torch.floor",
        ref="https://pytorch.org/docs/stable/generated/torch.floor.html",
    )
    @api_link(
        target="torch.Tensor.floor",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.floor.html",
    )
    def test_floor(self) -> None:
        for op, supports_out in [
            (torch.floor, True),
            (torch.Tensor.floor, False),
        ]:
            for input, expected in [
                (
                    torch.tensor([-0.3, -0.0, 0.0, 0.3, 1.2, 4.0], dtype=torch.float64),
                    torch.tensor([-1.0, -0.0, 0.0, 0.0, 1.0, 4.0], dtype=torch.float64),
                ),
                (
                    torch.tensor([-0.3, -0.0, 0.0, 0.3, 1.2, 4.0], dtype=torch.float32),
                    torch.tensor([-1.0, -0.0, 0.0, 0.0, 1.0, 4.0], dtype=torch.float32),
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op, input, expected=expected, supports_out=supports_out
                )

            for not_implemented in [
                torch.tensor(0, dtype=torch.int64),
                torch.tensor(False, dtype=torch.bool),
                torch.tensor(0.4 - 0.3j, dtype=torch.complex128),
            ]:
                assert_tensor_op_throws_not_implemented(op, not_implemented)

    @api_link(
        target="torch.clamp",
        ref="https://pytorch.org/docs/stable/generated/torch.clamp.html",
    )
    @api_link(
        target="torch.Tensor.clamp",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.clamp.html",
    )
    @api_link(
        target="torch.clip",
        ref="https://pytorch.org/docs/stable/generated/torch.clip.html",
        alias="torch.clamp",
    )
    @api_link(
        target="torch.Tensor.clamp",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.clip.html",
        alias="torch.Tensor.clamp",
    )
    def test_clamp(self) -> None:
        for op, supports_out in [
            (torch.clamp, True),
            (torch.clip, True),
            (torch.Tensor.clamp, False),
            (torch.Tensor.clip, False),
        ]:
            for input in [
                torch.tensor(0.3 + 2.0j, dtype=torch.complex128),
                torch.tensor(True, dtype=torch.bool),
            ]:
                assert_tensor_op_throws_not_implemented(
                    op,
                    input,
                    max=input,
                )

            for input in [
                torch.tensor([-0.3, -0.0, 0.0, 0.3, 1.2, 4.0], dtype=torch.float64),
                torch.tensor([0, 1, 8], dtype=torch.int64),
            ]:
                # using only min, only max, == to the source acts as identity.
                assert_cellwise_op_returns(
                    op,
                    input,
                    max=input,
                    expected=input,
                    supports_out=supports_out,
                )
                assert_cellwise_op_returns(
                    op,
                    input,
                    min=input,
                    expected=input,
                    supports_out=supports_out,
                )
                assert_cellwise_op_returns(
                    op,
                    input,
                    min=input,
                    max=input,
                    expected=input,
                    supports_out=supports_out,
                )

                assert_cellwise_op_returns(
                    op,
                    input + 2,
                    min=input - 5,
                    max=input + 5,
                    expected=input + 2,
                    supports_out=supports_out,
                )

                assert_cellwise_op_returns(
                    op,
                    input + 2,
                    max=input,
                    expected=input,
                    supports_out=supports_out,
                )
                assert_cellwise_op_returns(
                    op,
                    input + 2,
                    min=input,
                    expected=input + 2,
                    supports_out=supports_out,
                )

                assert_cellwise_op_returns(
                    op,
                    input - 2,
                    min=input,
                    expected=input,
                    supports_out=supports_out,
                )
                assert_cellwise_op_returns(
                    op,
                    input - 2,
                    max=input,
                    expected=input - 2,
                    supports_out=supports_out,
                )

    @api_link(
        target="torch.conj",
        ref="https://pytorch.org/docs/stable/generated/torch.conj.html",
    )
    @api_link(
        target="torch.Tensor.conj",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.conj.html",
    )
    @api_link(
        target="torch.conj_physical",
        ref="https://pytorch.org/docs/stable/generated/torch.conj_physical.html",
    )
    @api_link(
        target="torch.Tensor.conj_physical",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.conj_physical.html",
    )
    def test_conj_physical(self) -> None:
        for view_op, v_op_supports_out in [
            (torch.conj, False),
            (torch.Tensor.conj, False),
        ]:
            for copy_op, c_op_supports_out in [
                (torch.conj_physical, True),
                (torch.Tensor.conj_physical, False),
            ]:
                source = torch.tensor(
                    [-1 + 1j, -2 + 2j, 3 - 3j],
                    dtype=torch.complex128,
                )

                expected = torch.tensor(
                    [-1 - 1j, -2 - 2j, 3 + 3j],
                    dtype=torch.complex128,
                )

                assert_cellwise_op_returns(
                    view_op,
                    source,
                    expected=expected,
                    supports_out=v_op_supports_out,
                )

                assert_cellwise_op_returns(
                    copy_op,
                    source,
                    expected=expected,
                    supports_out=c_op_supports_out,
                )

                conj_copy = copy_op(source)  # type: ignore
                torch_eggs.assert_tensor_storage_differs(source, conj_copy)
                torch_eggs.assert_tensor_equals(
                    conj_copy,
                    expected,
                )

                conj_view = view_op(source)
                torch_eggs.assert_tensor_equals(
                    conj_view,
                    expected,
                    view_of=source,
                )

                torch_eggs.assert_tensor_equals(
                    conj_view.real,
                    source.real,
                    view_of=source,
                )
                torch_eggs.assert_tensor_equals(
                    conj_view.imag,
                    -source.imag,
                    view_of=source,
                )

                # Bidirectional conjugate writiable view.
                conj_view.imag[0] = 8

                torch_eggs.assert_tensor_equals(
                    source,
                    torch.tensor(
                        [-1 - 8j, -2 + 2j, 3 - 3j],
                        dtype=torch.complex128,
                    ),
                )

                torch_eggs.assert_tensor_equals(
                    conj_view,
                    torch.tensor(
                        [-1 + 8j, -2 - 2j, 3 + 3j],
                        dtype=torch.complex128,
                    ),
                )

    @api_link(
        target="torch.copysign",
        ref="https://pytorch.org/docs/stable/generated/torch.copysign.html",
    )
    @api_link(
        target="torch.Tensor.copysign",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.copysign.html",
    )
    def test_copysign(self) -> None:
        for op, supports_out in [
            (torch.copysign, True),
            (torch.Tensor.copysign, False),
        ]:

            for not_supported in [
                torch.tensor(0.3 + 2.0j, dtype=torch.complex64),
                torch.tensor(0.3 + 2.0j, dtype=torch.complex128),
            ]:
                assert_tensor_op_throws_not_implemented(
                    op,
                    not_supported,
                    torch.tensor(1),
                )

            for x, y, expected in [
                (
                    12.0,
                    -5.0,
                    -12.0,
                ),
                (
                    torch.tensor([-3, -4], dtype=torch.int64),
                    torch.tensor([2, -5], dtype=torch.int64),
                    # will convert from int to float.
                    torch.tensor([3.0, -4.0], dtype=torch.float32),
                ),
                (
                    torch.tensor([-3, -4], dtype=torch.float16),
                    torch.tensor([2, -5], dtype=torch.float16),
                    # will promote to the largest input dtype.
                    torch.tensor([3.0, -4.0], dtype=torch.float16),
                ),
                (
                    torch.tensor([-3, -4], dtype=torch.float32),
                    torch.tensor([2, -5], dtype=torch.float16),
                    # will promote to the largest input dtype.
                    torch.tensor([3.0, -4.0], dtype=torch.float32),
                ),
                (
                    torch.tensor([-3, -4], dtype=torch.int64),
                    # Bool signs sources get treated as ints { 0, 1 }.
                    torch.tensor([False, True]),
                    torch.tensor([3.0, 4.0], dtype=torch.float32),
                ),
                (
                    # broadcast
                    torch.tensor([-3, -4], dtype=torch.int64),
                    torch.tensor(
                        [
                            [[-1, -1], [-1, 1]],
                            [[1, -1], [1, 1]],
                        ],
                        dtype=torch.int64,
                    ),
                    # will convert from int to float.
                    torch.tensor(
                        [
                            [[-3.0, -4.0], [-3.0, 4.0]],
                            [[3.0, -4.0], [3.0, 4.0]],
                        ],
                        dtype=torch.float32,
                    ),
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op,
                    x,
                    y,
                    expected=expected,
                    supports_out=supports_out,
                )

    @api_link(
        target="torch.div",
        ref="https://pytorch.org/docs/stable/generated/torch.div.html",
    )
    @api_link(
        target="torch.Tensor.div",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.div.html",
    )
    @api_link(
        target="torch.divide",
        ref="https://pytorch.org/docs/stable/generated/torch.divide.html",
        alias="torch.div",
    )
    @api_link(
        target="torch.Tensor.divide",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.divide.html",
        alias="torch.Tensor.div",
    )
    def test_div(self) -> None:
        for op, supports_out in [
            (torch.div, True),
            (torch.divide, True),
            (torch.Tensor.div, False),
            (torch.Tensor.divide, False),
        ]:
            for x, y, expected in [
                (
                    [10, 20],
                    4,
                    [2.5, 5.0],
                ),
                (
                    [10 + 10j, 20 + 20j],
                    [4, 4j],
                    [2.5 + 2.5j, 5.0 - 5j],
                ),
                (
                    [False, False, True, True],
                    [False, True, False, True],
                    [torch.nan, 0.0, torch.inf, 1.0],
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op,
                    x,
                    y,
                    expected=expected,
                    supports_out=supports_out,
                )

            # "trunc" depends on out.trunc() being supported.
            for not_supported in [
                0.3 + 2.0j,
            ]:
                assert_tensor_op_throws_not_implemented(
                    op,
                    not_supported,
                    torch.tensor(1),
                    rounding_mode="trunc",
                )

            for x, y, expected in [
                (
                    [10, 20, -10, -20],
                    4,
                    [2, 5, -2, -5],
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op,
                    x,
                    y,
                    rounding_mode="trunc",
                    expected=expected,
                    supports_out=supports_out,
                )

            # "floor" depends on out.floor() being supported.
            for not_supported in [
                0.3 + 2.0j,
            ]:
                assert_tensor_op_throws_not_implemented(
                    op,
                    not_supported,
                    torch.tensor(1),
                    rounding_mode="floor",
                )

            for x, y, expected in [
                (
                    [10, 20, -10, -20],
                    4,
                    [2, 5, -3, -5],
                ),
            ]:
                assert_cellwise_bin_op_returns(
                    op,
                    x,
                    y,
                    rounding_mode="floor",
                    expected=expected,
                    supports_out=supports_out,
                )
