from collections import namedtuple
import typing
from typing import Callable
import unittest

import hamcrest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs
from smot.testlib.torch_eggs import TensorConvertable, assert_tensor_equals


def assert_tensor_uniop_unsupported(
    op: Callable[[torch.Tensor], torch.Tensor],
    source: TensorConvertable,
) -> None:
    t_source: torch.Tensor = torch.as_tensor(source)
    eggs.assert_raises(
        lambda: op(t_source),
        RuntimeError,
        "not implemented",
    )


def assert_tensor_uniop(
    op: Callable[[torch.Tensor], torch.Tensor],
    source: TensorConvertable,
    expected: TensorConvertable,
    *,
    close: bool = False,
    supports_out: bool = True,
) -> None:
    t_source = torch.as_tensor(source)
    t_expected = torch.as_tensor(expected)

    result = op(t_source)
    assert_tensor_equals(
        result,
        t_expected,
        close=close,
    )

    # use the shape of expected to build an out.
    out = torch.empty_like(result)

    if supports_out:
        eggs.assert_match(
            op(t_source, out=out),  # type: ignore
            hamcrest.same_instance(out),
        )
        assert_tensor_equals(
            out,
            t_expected,
            close=close,
        )

    else:
        eggs.assert_raises(
            lambda: op(t_source, out=out),  # type: ignore
            TypeError,
        )


def assert_tensor_uniop_pair(
    torch_op: Callable[[torch.Tensor], torch.Tensor],
    tensor_op: Callable[[torch.Tensor], torch.Tensor],
    source: TensorConvertable,
    expected: TensorConvertable,
    close: bool = False,
) -> None:
    source = torch.as_tensor(source)
    expected = torch.as_tensor(expected)

    assert_tensor_uniop(
        torch_op,
        source,
        expected,
        close=close,
    )
    assert_tensor_uniop(
        tensor_op,
        source,
        expected,
        close=close,
        supports_out=False,
    )


UniOpExample = namedtuple("UniOpExample", ("input", "expected"))


def assert_tensor_uniop_pair_cases(
    *,
    torch_op: Callable[[torch.Tensor], torch.Tensor],
    tensor_op: Callable[[torch.Tensor], torch.Tensor],
    unsupported: typing.Optional[
        typing.Sequence[
            TensorConvertable,
        ]
    ] = None,
    close: bool = False,
    cases: typing.Sequence[UniOpExample],
) -> None:
    for source, expected in cases:
        assert_tensor_uniop_pair(
            torch_op,
            tensor_op,
            source,
            expected,
            close=close,
        )

    if unsupported:
        for source in unsupported:
            source = torch.as_tensor(source)

            assert_tensor_uniop_unsupported(
                torch_op,
                source,
            )
            assert_tensor_uniop_unsupported(
                tensor_op,
                source,
            )


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
            assert_tensor_uniop_pair_cases(
                torch_op=op,
                tensor_op=bound_op,
                unsupported=[
                    [True, False],
                ],
                cases=[
                    UniOpExample(
                        input=[],
                        expected=[],
                    ),
                    UniOpExample(
                        input=torch.nan,
                        expected=torch.nan,
                    ),
                    UniOpExample(
                        input=-3,
                        expected=3,
                    ),
                    UniOpExample(
                        input=[[-1], [3]],
                        expected=[[1], [3]],
                    ),
                    UniOpExample(
                        input=[[-1.5], [3.5]],
                        expected=[[1.5], [3.5]],
                    ),
                    UniOpExample(
                        input=[[-1.5 + 0j], [-3 + 4j]],
                        expected=[[1.5], [5.0]],
                    ),
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
            assert_tensor_uniop_pair_cases(
                torch_op=op,
                tensor_op=bound_op,
                close=True,
                cases=[
                    UniOpExample(
                        input=[],
                        expected=[],
                    ),
                    UniOpExample(
                        input=torch.nan,
                        expected=torch.nan,
                    ),
                    UniOpExample(
                        input=-3,
                        expected=torch.nan,
                    ),
                    UniOpExample(
                        input=[[0], [1], [-1], [3]],
                        expected=[[1.5707963705], [0], [torch.pi], [torch.nan]],
                    ),
                    UniOpExample(
                        input=[[0.0], [1.0], [-1.0], [3.0]],
                        expected=[[1.5707963705], [0], [torch.pi], [torch.nan]],
                    ),
                    UniOpExample(
                        input=[True, False],
                        expected=[0.0, 1.5707963705],
                    ),
                    UniOpExample(
                        input=[0j, 1 + 1j],
                        expected=[
                            1.5707963705 - 0.0000000000j,
                            0.9045568705 - 1.0612751245j,
                        ],
                    ),
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
            assert_tensor_uniop_pair_cases(
                torch_op=op,
                tensor_op=bound_op,
                close=True,
                cases=[
                    UniOpExample(
                        input=[],
                        expected=[],
                    ),
                    UniOpExample(
                        input=torch.nan,
                        expected=torch.nan,
                    ),
                    UniOpExample(
                        input=-3,
                        expected=torch.nan,
                    ),
                    UniOpExample(
                        # int input
                        input=[[0], [1], [-1], [3]],
                        expected=[
                            [0.0000000000],
                            [1.5707963705],
                            [-1.5707963705],
                            [torch.nan],
                        ],
                    ),
                    UniOpExample(
                        # float input
                        input=[[0.0], [1.0], [-1.0], [3.0]],
                        expected=[
                            [0.0000000000],
                            [1.5707963705],
                            [-1.5707963705],
                            [torch.nan],
                        ],
                    ),
                    UniOpExample(
                        input=[True, False],
                        expected=[1.5707963705, 0.0],
                    ),
                    UniOpExample(
                        input=[0j, 1 + 1j],
                        expected=[0.0j, 0.6662394404 + 1.0612752438j],
                    ),
                ],
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
            assert_tensor_uniop_pair_cases(
                torch_op=op,
                tensor_op=bound_op,
                close=True,
                cases=[
                    UniOpExample(
                        input=[],
                        expected=[],
                    ),
                    UniOpExample(
                        input=torch.nan,
                        expected=torch.nan,
                    ),
                    UniOpExample(
                        input=-3,
                        expected=torch.nan,
                    ),
                    UniOpExample(
                        input=2,
                        expected=1.31695795,
                    ),
                    UniOpExample(
                        input=[[0], [1], [torch.pi], [2]],
                        expected=[[torch.nan], [0.0], [1.8115262], [1.31695795]],
                    ),
                    UniOpExample(
                        input=[True, False],
                        expected=[0.0, torch.nan],
                    ),
                    UniOpExample(
                        input=[0j, 1 + 1j],
                        expected=[
                            0.0000000000 + 1.5707963705j,
                            1.0612751245 + 0.9045568705j,
                        ],
                    ),
                ],
            )
