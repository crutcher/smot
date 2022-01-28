from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass
class PairedOpChecker:
    torch_op: Callable[[torch.Tensor], torch.Tensor]
    tensor_op: Callable[[torch.Tensor], torch.Tensor]

    def __enter__(self) -> "PairedOpChecker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def assert_case_returns(
        self,
        input: Any,
        *,
        returns: Any,
        close: bool = False,
    ) -> None:
        input = torch.as_tensor(input)
        returns = torch.as_tensor(returns)

        assert_tensor_uniop(
            self.torch_op,
            input,
            returns,
            close=close,
        )
        assert_tensor_uniop(
            self.tensor_op,
            input,
            returns,
            close=close,
            supports_out=False,
        )

    def assert_case_not_implemented(
        self,
        input: Any,
    ) -> None:
        source = torch.as_tensor(input)

        assert_tensor_uniop_not_implemented(
            self.torch_op,
            source,
        )
        assert_tensor_uniop_not_implemented(
            self.tensor_op,
            source,
        )
