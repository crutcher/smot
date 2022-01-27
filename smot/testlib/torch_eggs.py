import contextlib
import numbers
import typing
from typing import Callable, Tuple

import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
import nptyping
import torch

from smot.testlib import eggs


def hide_tracebacks(module: typing.Any, mode: bool) -> None:
    # TODO: lift to eggs.
    # unittest integration; hide these frames from tracebacks
    module["__unittest"] = mode
    # py.test integration; hide these frames from tracebacks
    module["__tracebackhide__"] = mode


# hide by default.
hide_tracebacks(globals(), True)


def assert_views(source: torch.Tensor, *tensor: torch.Tensor) -> None:
    for t in tensor:
        eggs.assert_match(
            t.storage().data_ptr(),  # type: ignore
            source.storage().data_ptr(),  # type: ignore
        )


def assert_not_view(tensor: torch.Tensor, source: torch.Tensor) -> None:
    eggs.assert_match(
        tensor.storage().data_ptr(),  # type: ignore
        hamcrest.not_(source.storage().data_ptr()),  # type: ignore
    )


TensorConvertable = typing.Union[
    torch.Tensor,
    numbers.Number,
    typing.Sequence,
    nptyping.NDArray,
]


class TensorStructureMatcher(BaseMatcher):
    expected: torch.Tensor

    def __init__(self, expected: TensorConvertable) -> None:
        self.expected = torch.as_tensor(expected)

    def _matches(self, item: typing.Any) -> bool:
        # Todo: structural miss-match that still shows expected tensor.

        try:
            eggs.assert_match(
                item.device,
                self.expected.device,
            )
            eggs.assert_match(
                item.size(),
                self.expected.size(),
            )
            eggs.assert_match(
                item.dtype,
                self.expected.dtype,
            )
            eggs.assert_match(
                item.layout,
                self.expected.layout,
            )
            return True
        except AssertionError:
            return False

    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.expected)


def expect_tensor_structure(
    expected: TensorConvertable,
) -> TensorStructureMatcher:
    return TensorStructureMatcher(expected)


def assert_tensor_structure(
    actual: torch.Tensor,
    expected: TensorConvertable,
) -> None:
    hamcrest.assert_that(
        actual,
        expect_tensor_structure(expected),
    )


class TensorMatcher(TensorStructureMatcher):
    close: bool = False

    def __init__(
        self,
        expected: TensorConvertable,
        *,
        close: bool = False,
    ):
        super().__init__(expected=expected)
        self.close = close

        if self.expected.is_sparse and not self.expected.is_coalesced():
            self.expected = self.expected.coalesce()

    def _matches(self, item: typing.Any) -> bool:
        if not super()._matches(item):
            return False

        if self.close:
            torch.testing.assert_close(
                item,
                self.expected,
                equal_nan=True,
            )
            return True

        else:
            if self.expected.is_sparse:
                eggs.assert_true(item.is_sparse)

                # TODO: it may be necessary to sort the indices and values.
                if not item.is_coalesced():
                    item = item.coalesce()

                assert_tensor(
                    item.indices(),
                    self.expected.indices(),
                )
                assert_tensor(
                    item.values(),
                    self.expected.values(),
                )
                return True
            else:
                return torch.equal(item, self.expected)

    def describe_to(self, description: Description) -> None:
        description.append_text("\n")
        description.append_description_of(self.expected)

    def describe_match(self, item: typing.Any, match_description: Description) -> None:
        match_description.append_text("was \n")
        match_description.append_description_of(item)

    def describe_mismatch(
        self, item: typing.Any, mismatch_description: Description
    ) -> None:
        mismatch_description.append_text("was \n")
        mismatch_description.append_description_of(item)


def expect_tensor(
    expected: TensorConvertable,
) -> TensorMatcher:
    return TensorMatcher(expected, close=False)


def expect_tensor_seq(
    *expected: TensorConvertable,
) -> Matcher:
    return hamcrest.contains_exactly(*[expect_tensor(e) for e in expected])


def assert_tensor(
    actual: torch.Tensor,
    expected: TensorConvertable,
) -> None:
    hamcrest.assert_that(
        actual,
        expect_tensor(expected),
    )


def assert_view_tensor(
    actual: torch.Tensor,
    source: torch.Tensor,
    expected: TensorConvertable,
) -> None:
    assert_views(source, actual)
    assert_tensor(actual, expected)


def assert_tensor_seq(
    actual: typing.Sequence[torch.Tensor],
    *expected: TensorConvertable,
) -> None:
    hamcrest.assert_that(
        actual,
        expect_tensor_seq(*expected),
    )


def assert_view_tensor_seq(
    actual: typing.Sequence[torch.Tensor],
    source: torch.Tensor,
    *expected: TensorConvertable,
) -> None:
    assert_views(source, *actual)
    hamcrest.assert_that(
        actual,
        expect_tensor_seq(*expected),
    )


def expect_tensor_close(
    expected: TensorConvertable,
) -> TensorMatcher:
    return TensorMatcher(expected, close=True)


def assert_tensor_close(
    actual: torch.Tensor,
    expected: TensorConvertable,
) -> None:
    hamcrest.assert_that(
        actual,
        expect_tensor_close(expected),
    )


@contextlib.contextmanager
def reset_generator_seed(seed: int = 3 * 17 * 53 + 1) -> typing.Iterator:
    torch.manual_seed(seed)
    yield


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
    supports_out: bool = True,
) -> None:
    t_source = torch.as_tensor(source)
    t_expected = torch.as_tensor(expected)


    result = op(t_source)
    assert_tensor(
        result,
        t_expected,
    )

    out = torch.empty_like(result)

    if supports_out:
        # use the shape of expected to build an out.
        eggs.assert_match(
            op(t_source, out=out),  # type: ignore
            hamcrest.same_instance(out),
        )
        assert_tensor(
            out,
            t_expected,
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
) -> None:
    source = torch.as_tensor(source)
    expected = torch.as_tensor(expected)

    assert_tensor_uniop(
        torch_op,
        source,
        expected,
    )
    assert_tensor_uniop(
        tensor_op,
        source,
        expected,
        supports_out=False,
    )


def assert_tensor_uniop_pair_cases(
    torch_op: Callable[[torch.Tensor], torch.Tensor],
    tensor_op: Callable[[torch.Tensor], torch.Tensor],
    *cases: Tuple[
        TensorConvertable,
        TensorConvertable,
    ],
    unsupported: typing.Optional[
        typing.Sequence[
            TensorConvertable,
        ]
    ] = None,
) -> None:
    for source, expected in cases:
        assert_tensor_uniop_pair(
            torch_op,
            tensor_op,
            source,
            expected,
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
