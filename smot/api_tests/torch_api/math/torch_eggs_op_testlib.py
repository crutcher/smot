from typing import Any, Callable, Tuple, Union

import hamcrest
import torch

from smot.testlib import eggs
from smot.testlib.torch_eggs import TensorConvertable, assert_tensor_equals


def hide_tracebacks(mode: bool = True) -> None:
    """
    Hint that some unittest stacks (unittest, pytest) should remove
    frames from tracebacks that include this module.

    :param mode: optional, the traceback mode.
    """
    eggs.hide_module_tracebacks(globals(), mode)


# hide by default.
hide_tracebacks(True)


def assert_tensor_uniop_not_implemented(
    op: Union[Callable[[torch.Tensor], torch.Tensor], Any],
    source: TensorConvertable,
) -> None:
    t_source: torch.Tensor = torch.as_tensor(source)
    eggs.assert_raises(
        lambda: op(t_source),
        RuntimeError,
        r"not (implemented|supported)",
    )


def _assert_cellwise_op_returns(
    op: Union[Callable[[torch.Tensor], torch.Tensor], Any],
    input: Tuple[torch.Tensor, ...],
    expected: TensorConvertable,
    *,
    close: bool = False,
    supports_out: bool = True,
) -> None:
    """
    Assert that the given op is a well behaving cell-wise unitary operation.

    :param op: the operation to test.
    :param input: the input.
    :param expected: the expected result.
    :param close: should the expected result be evaluated as "close" or exact?
    :param supports_out: does this operation support the `out` keyword?
    """
    t_expected = torch.as_tensor(expected)

    result = op(*input)
    assert_tensor_equals(
        result,
        t_expected,
        close=close,
    )

    # check structural transforms work.
    tile_pattern = (2, 3, 1)
    tiled_input = tuple(torch.tile(x, tile_pattern) for x in input)
    tiled_expected = torch.tile(t_expected, tile_pattern)
    assert_tensor_equals(
        op(*tiled_input),
        tiled_expected,
        close=close,
    )

    # use the shape of expected to build an out.
    out = torch.empty_like(result)

    if supports_out:
        eggs.assert_match(
            op(*input, out=out),  # type: ignore
            hamcrest.same_instance(out),
        )
        assert_tensor_equals(
            out,
            t_expected,
            close=close,
        )

    else:
        eggs.assert_raises(
            lambda: op(*input, out=out),  # type: ignore
            TypeError,
        )


def assert_cellwise_unary_op_returns(
    op: Union[Callable[[torch.Tensor], torch.Tensor], Any],
    input: TensorConvertable,
    expected: TensorConvertable,
    *,
    close: bool = False,
    supports_out: bool = True,
) -> None:
    """
    Assert that the given op is a well behaving cell-wise unitary operation.

    :param op: the operation to test.
    :param input: the input.
    :param expected: the expected result.
    :param close: should the expected result be evaluated as "close" or exact?
    :param supports_out: does this operation support the `out` keyword?
    """
    _assert_cellwise_op_returns(
        op=op,
        input=(torch.as_tensor(input),),
        expected=expected,
        close=close,
        supports_out=supports_out,
    )


def assert_cellwise_bin_op_returns(
    op: Union[Callable[[torch.Tensor], torch.Tensor], Any],
    input: TensorConvertable,
    other: TensorConvertable,
    expected: TensorConvertable,
    *,
    close: bool = False,
    supports_out: bool = True,
) -> None:
    """
    Assert that the given op is a well behaving cell-wise unitary operation.

    :param op: the operation to test.
    :param input: the input.
    :param other: the other input.
    :param expected: the expected result.
    :param close: should the expected result be evaluated as "close" or exact?
    :param supports_out: does this operation support the `out` keyword?
    """
    _assert_cellwise_op_returns(
        op=op,
        input=(torch.as_tensor(input), torch.as_tensor(other)),
        expected=expected,
        close=close,
        supports_out=supports_out,
    )
