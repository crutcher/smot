from typing import Any, Callable, List, Type, Union

import hamcrest
import torch

from smot.testlib import eggs
from smot.testlib.torch_eggs import assert_tensor_equals


def hide_tracebacks(mode: bool = True) -> None:
    """
    Hint that some unittest stacks (unittest, pytest) should remove
    frames from tracebacks that include this module.

    :param mode: optional, the traceback mode.
    """
    eggs.hide_module_tracebacks(globals(), mode)


# hide by default.
hide_tracebacks(False)


def assert_tensor_op_throws(
    op: Union[Callable[[torch.Tensor], torch.Tensor], Any],
    *args: Any,
    exception_type: Type[Exception],
    exception_pattern: str,
    **kwargs: Any,
) -> None:
    t_source: List[torch.Tensor] = [torch.as_tensor(a) for a in args]
    eggs.assert_raises(
        lambda: op(*t_source, **kwargs),
        exception_type,
        exception_pattern,
    )


def assert_tensor_op_throws_not_implemented(
    op: Union[Callable[[torch.Tensor], torch.Tensor], Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    assert_tensor_op_throws(
        op,
        *args,
        exception_type=RuntimeError,
        exception_pattern=r"not (implemented|supported)",
        **kwargs,
    )


def assert_cellwise_op_returns(
    op: Union[Callable[[torch.Tensor], torch.Tensor], Any],
    *args: torch.Tensor,
    expected: Any,
    close: bool = False,
    supports_out: bool = True,
    **kwargs: Any,
) -> None:
    """
    Assert that the given op is a well behaving cell-wise unitary operation.

    :param op: the operation to test.
    :param args: the input.
    :param expected: the expected result.
    :param close: should the expected result be evaluated as "close" or exact?
    :param supports_out: does this operation support the `out` keyword?
    """
    t_expected = torch.as_tensor(expected)

    if "args" in kwargs:
        args = kwargs["args"]
        del kwargs["args"]

    result = op(*args, **kwargs)
    assert_tensor_equals(
        result,
        t_expected,
        close=close,
    )

    # check structural transforms work.
    # tile_pattern = (3, 2)
    # tiled_input = tuple(torch.tile(x, tile_pattern) for x in args)
    # tiled_expected = torch.tile(t_expected, tile_pattern)
    # assert_tensor_equals(
    #     op(*tiled_input, **kwargs),
    #     tiled_expected,
    #     close=close,
    # )

    # use the shape of expected to build an out.
    out = torch.empty_like(result)

    if supports_out:
        eggs.assert_match(
            op(*args, out=out, **kwargs),  # type: ignore
            hamcrest.same_instance(out),
        )
        assert_tensor_equals(
            out,
            t_expected,
            close=close,
        )

    else:
        eggs.assert_raises(
            lambda: op(*args, out=out, **kwargs),  # type: ignore
            TypeError,
        )


def assert_cellwise_unary_op_returns(
    op: Union[Callable[[torch.Tensor], torch.Tensor], Any],
    input: Any,
    *,
    expected: Any,
    close: bool = False,
    supports_out: bool = True,
    **kwargs: Any,
) -> None:
    """
    Assert that the given op is a well behaving cell-wise unitary operation.

    :param op: the operation to test.
    :param input: the input.
    :param expected: the expected result.
    :param close: should the expected result be evaluated as "close" or exact?
    :param supports_out: does this operation support the `out` keyword?
    """
    assert_cellwise_op_returns(
        op,
        *(torch.as_tensor(input),),
        expected=expected,
        close=close,
        supports_out=supports_out,
        **kwargs,
    )


def assert_cellwise_bin_op_returns(
    op: Union[Callable[[torch.Tensor], torch.Tensor], Any],
    input: Any,
    other: Any,
    *,
    expected: Any,
    close: bool = False,
    supports_out: bool = True,
    **kwargs: Any,
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
    assert_cellwise_op_returns(
        op,
        *(torch.as_tensor(input), torch.as_tensor(other)),
        expected=expected,
        close=close,
        supports_out=supports_out,
        **kwargs,
    )
