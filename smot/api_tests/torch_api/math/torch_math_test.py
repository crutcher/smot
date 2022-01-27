import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


class MathTest(unittest.TestCase):
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
            torch_eggs.assert_tensor_uniop_pair_cases(
                op,
                bound_op,
                # ints
                (
                    [],
                    [],
                ),
                (
                    torch.nan,
                    torch.nan,
                ),
                (
                    -3,
                    3,
                ),
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
            torch_eggs.assert_tensor_uniop_pair_cases(
                op,
                bound_op,
                (
                    [],
                    [],
                ),
                (
                    torch.nan,
                    torch.nan,
                ),
                (
                    -3,
                    torch.nan,
                ),
                (
                    [[0], [1], [-1], [3]],
                    [[1.5707963705], [0], [torch.pi], [torch.nan]],
                ),
                (
                    [[0.0], [1.0], [-1.0], [3.0]],
                    [[1.5707963705], [0], [torch.pi], [torch.nan]],
                ),
                # bools, are cast to ints  ...
                (
                    [True, False],
                    [0.0, 1.5707963705],
                ),
                # complex
                (
                    [0j, 1 + 1j],
                    [1.5707963705 - 0.0000000000j, 0.9045568705 - 1.0612751245j],
                ),
                close=True,
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
            torch_eggs.assert_tensor_uniop_pair_cases(
                op,
                bound_op,
                (
                    [],
                    [],
                ),
                (
                    torch.nan,
                    torch.nan,
                ),
                (
                    -3,
                    torch.nan,
                ),
                (
                    2,
                    1.31695795,
                ),
                (
                    [[0], [1], [torch.pi], [2]],
                    [[torch.nan], [0.0], [1.8115262], [1.31695795]],
                ),
                # bools, are cast to ints  ...
                (
                    [True, False],
                    [0.0, torch.nan],
                ),
                # complex
                (
                    [0j, 1 + 1j],
                    [0.0000000000+1.5707963705j, 1.0612751245+0.9045568705j],
                ),
                close=True,
            )
