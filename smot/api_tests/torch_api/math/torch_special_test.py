import unittest

import numpy as np
import torch

from smot.api_tests.torch_api.math.torch_eggs_op_testlib import (
    assert_cellwise_unary_op_returns,
    assert_tensor_op_throws_not_implemented,
)
from smot.doc_link.link_annotations import api_link


class TorchSpecialTest(unittest.TestCase):
    @api_link(
        target="torch.special.entr",
        ref="https://pytorch.org/docs/stable/generated/special.html#torch.special.entr",
    )
    def test_entr(self) -> None:
        for op, supports_out in [
            (torch.special.entr, True),
        ]:
            for input, expected in [
                (
                    [False, True],
                    torch.tensor(
                        [0.0, -np.log(1.0)],
                        dtype=torch.float32,
                    ),
                ),
                (
                    [-1.0, 0.0, 0.5, 1.0],
                    torch.tensor(
                        [-torch.inf, 0.0, -0.5 * np.log(0.5), -np.log(1.0)],
                        dtype=torch.float32,
                    ),
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected=expected,
                    close=True,
                    supports_out=supports_out,
                )

            for not_implemented in [
                [0 + 0j, 1 + 1j, 0.5 - 0.5j],
            ]:
                assert_tensor_op_throws_not_implemented(op, not_implemented)

    @api_link(
        target="torch.erf",
        ref="https://pytorch.org/docs/stable/generated/torch.erf.html",
    )
    @api_link(
        target="torch.erfc",
        ref="https://pytorch.org/docs/stable/generated/torch.erfc.html",
    )
    @api_link(
        target="torch.special.erf",
        ref="https://pytorch.org/docs/stable/generated/special.html#torch.special.erf",
    )
    @api_link(
        target="torch.special.erfc",
        ref="https://pytorch.org/docs/stable/generated/special.html#torch.special.erfc",
    )
    @api_link(
        target="torch.Tensor.erf",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.erf.html",
    )
    @api_link(
        target="torch.Tensor.erfc",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.erfc.html",
    )
    def test_erf_erfinv_erfc(self) -> None:
        for erf, erfinv, erfc, supports_out in [
            (torch.erf, torch.erfinv, torch.erfc, True),
            (torch.special.erf, torch.special.erfinv, torch.special.erfc, True),
            (torch.Tensor.erf, torch.Tensor.erfinv, torch.Tensor.erfc, False),
        ]:
            for input, expected in [
                (
                    [False, True],
                    torch.tensor(
                        [0.0, 0.8427007],
                        dtype=torch.float32,
                    ),
                ),
                (
                    [-1.0, 0.0, 0.5, 1.0],
                    torch.tensor(
                        [-0.8427007, 0.0, 0.5204998851, 0.8427007794],
                        dtype=torch.float32,
                    ),
                ),
            ]:
                input = torch.as_tensor(input)
                expected = torch.as_tensor(expected)

                assert_cellwise_unary_op_returns(
                    erf,
                    input,
                    expected=expected,
                    close=True,
                    supports_out=supports_out,
                )

                # erfinv(erf(x)) == x
                assert_cellwise_unary_op_returns(
                    erfinv,
                    expected,
                    expected=input.to(expected.dtype),
                    close=True,
                    supports_out=supports_out,
                )

                assert_cellwise_unary_op_returns(
                    erfc,
                    input,
                    expected=1.0 - expected,
                    close=True,
                    supports_out=supports_out,
                )

            for not_implemented in [
                [0 + 0j, 1 + 1j, 0.5 - 0.5j],
            ]:
                assert_tensor_op_throws_not_implemented(erf, not_implemented)
                assert_tensor_op_throws_not_implemented(erfc, not_implemented)

    @api_link(
        target="torch.special.erfcx",
        ref="https://pytorch.org/docs/stable/generated/special.html#torch.special.erfcx",
    )
    def test_erfcx(self) -> None:
        for erfcx in [
            torch.special.erfcx,
        ]:
            for input, expected in [
                (
                    torch.tensor([0.0, -1.0, 10.0], dtype=torch.float64),
                    torch.tensor([1.0, 5.00898, 0.05614099], dtype=torch.float64),
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    erfcx,
                    input,
                    expected=expected,
                    close=True,
                    supports_out=True,
                )

                # erfcx(x) := exp(x**2) * erfc(x)
                assert_cellwise_unary_op_returns(
                    erfcx,
                    input,
                    expected=torch.exp(input**2) * torch.special.erfc(input),
                    close=True,
                    supports_out=True,
                )

            for not_implemented in [
                [0 + 0j, 1 + 1j, 0.5 - 0.5j],
            ]:
                assert_tensor_op_throws_not_implemented(erfcx, not_implemented)

    @api_link(
        target="torch.digamma",
        ref="https://pytorch.org/docs/stable/generated/torch.abs.html",
        alias="torch.special.digamma",
    )
    @api_link(
        target="torch.special.digamma",
        ref="https://pytorch.org/docs/stable/generated/special.html#torch.special.digamma",
    )
    @api_link(
        target="torch.Tensor.digamma",
        ref="https://pytorch.org/docs/stable/generated/torch.Tensor.digamma.html",
    )
    def test_digamma(self) -> None:
        for op, supports_out in [
            (torch.digamma, True),
            (torch.special.digamma, True),
            (torch.Tensor.digamma, False),
        ]:
            for input, expected in [
                (
                    # From PyTorch 1.8 onwards, the digamma function returns -Inf for 0.
                    # Previously it returned NaN for 0.
                    0,
                    -torch.inf,
                ),
                (
                    [False, True],
                    [-torch.inf, -0.57721591],
                ),
                (
                    [0, 1, 0.5],
                    [-torch.inf, -0.57721591, -1.9635108],
                ),
            ]:
                assert_cellwise_unary_op_returns(
                    op,
                    input,
                    expected=expected,
                    close=True,
                    supports_out=supports_out,
                )

            for not_implemented in [
                [0 + 0j, 1 + 1j, 0.5 - 0.5j],
            ]:
                assert_tensor_op_throws_not_implemented(op, not_implemented)
