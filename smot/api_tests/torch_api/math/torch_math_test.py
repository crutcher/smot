import unittest

import torch

from smot.api_tests.torch_api.math.torch_eggs_op_testlib import PairedOpChecker
from smot.doc_link.link_annotations import api_link


class TrigMathTest(unittest.TestCase):
    @api_link(
        target="torch.abs",
        ref="https://pytorch.org/docs/stable/generated/torch.abs.html",
    )
    @api_link(
        target="torch.absolute",
        ref="https://pytorch.org/docs/stable/generated/torch.absolute.html",
        alias="torch.abs",
    )
    def test_abs(self) -> None:
        for op, bound_op in [
            (torch.abs, torch.Tensor.abs),
            (torch.absolute, torch.Tensor.absolute),
        ]:
            with PairedOpChecker(op, bound_op) as checker:
                checker.assert_case_returns(
                    [],
                    returns=[],
                )

                checker.assert_case_returns(
                    [],
                    returns=[],
                )

                checker.assert_case_returns(
                    torch.nan,
                    returns=torch.nan,
                )

                checker.assert_case_returns(
                    -3,
                    returns=3,
                )

                checker.assert_case_returns(
                    [[-1], [3]],
                    returns=[[1], [3]],
                )

                checker.assert_case_returns(
                    [[-1.5], [3.5]],
                    returns=[[1.5], [3.5]],
                )

                checker.assert_case_returns(
                    [[-1.5 + 0j], [-3 + 4j]],
                    returns=[[1.5], [5.0]],
                )

                checker.assert_case_not_implemented(
                    [True, False],
                )

    @api_link(
        target="torch.acos",
        ref="https://pytorch.org/docs/stable/generated/torch.acos.html",
    )
    @api_link(
        target="torch.arccos",
        ref="https://pytorch.org/docs/stable/generated/torch.arccos.html",
        alias="torch.acos",
    )
    def test_acos(self) -> None:
        for op, bound_op in [
            (torch.acos, torch.Tensor.acos),
            (torch.arccos, torch.Tensor.arccos),
        ]:
            with PairedOpChecker(op, bound_op) as checker:
                checker.assert_case_returns(
                    [],
                    returns=[],
                )
                checker.assert_case_returns(
                    torch.nan,
                    returns=torch.nan,
                )
                checker.assert_case_returns(
                    -3,
                    returns=torch.nan,
                )
                checker.assert_case_returns(
                    [[0], [1], [-1], [3]],
                    returns=[[1.5707963705], [0], [torch.pi], [torch.nan]],
                )
                checker.assert_case_returns(
                    [[0.0], [1.0], [-1.0], [3.0]],
                    returns=[[1.5707963705], [0], [torch.pi], [torch.nan]],
                )
                checker.assert_case_returns(
                    [True, False],
                    returns=[0.0, 1.5707963705],
                )
                checker.assert_case_returns(
                    [0j, 1 + 1j],
                    returns=[
                        1.5707963705 - 0.0000000000j,
                        0.9045568705 - 1.0612751245j,
                    ],
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
    def test_asin(self) -> None:
        for op, bound_op in [
            (torch.asin, torch.Tensor.asin),
            (torch.arcsin, torch.Tensor.arcsin),
        ]:
            with PairedOpChecker(op, bound_op) as checker:
                checker.assert_case_returns(
                    [],
                    returns=[],
                )
                checker.assert_case_returns(
                    [],
                    returns=[],
                )
                checker.assert_case_returns(
                    torch.nan,
                    returns=torch.nan,
                )
                checker.assert_case_returns(
                    -3,
                    returns=torch.nan,
                )
                checker.assert_case_returns(
                    # int input
                    [[0], [1], [-1], [3]],
                    returns=[
                        [0.0000000000],
                        [1.5707963705],
                        [-1.5707963705],
                        [torch.nan],
                    ],
                )
                checker.assert_case_returns(
                    # float input
                    [[0.0], [1.0], [-1.0], [3.0]],
                    returns=[
                        [0.0000000000],
                        [1.5707963705],
                        [-1.5707963705],
                        [torch.nan],
                    ],
                )
                checker.assert_case_returns(
                    [True, False],
                    returns=[1.5707963705, 0.0],
                )
                checker.assert_case_returns(
                    [0j, 1 + 1j],
                    returns=[0.0j, 0.6662394404 + 1.0612752438j],
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
    def test_acosh(self) -> None:
        for op, bound_op in [
            (torch.acosh, torch.Tensor.acosh),
            (torch.arccosh, torch.Tensor.arccosh),
        ]:
            with PairedOpChecker(op, bound_op) as checker:
                checker.assert_case_returns(
                    [],
                    returns=[],
                )
                checker.assert_case_returns(
                    torch.nan,
                    returns=torch.nan,
                )
                checker.assert_case_returns(
                    -3,
                    returns=torch.nan,
                )
                checker.assert_case_returns(
                    2,
                    returns=1.31695795,
                )
                checker.assert_case_returns(
                    [
                        [0],
                        [1],
                        [torch.pi],
                        [2],
                    ],
                    returns=[
                        [torch.nan],
                        [0.0],
                        [1.8115262],
                        [1.31695795],
                    ],
                    close=True,
                )
                checker.assert_case_returns(
                    [True, False],
                    returns=[0.0, torch.nan],
                )
                checker.assert_case_returns(
                    [0j, 1 + 1j],
                    returns=[
                        0.0000000000 + 1.5707963705j,
                        1.0612751245 + 0.9045568705j,
                    ],
                    close=True,
                )
