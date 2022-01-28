import contextlib
import numbers
import typing
from typing import Callable, Tuple

import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
import nptyping
from overrides import overrides
import torch

from smot.testlib import eggs
from smot.testlib.eggs import hide_module_tracebacks

TensorConvertable = typing.Union[
    torch.Tensor,
    numbers.Number,
    typing.Sequence,
    nptyping.NDArray,
]
"Types which torch.as_tensor(T) can convert."


def hide_tracebacks(mode: bool = True) -> None:
    """
    Hint that some unittest stacks (unittest, pytest) should remove
    frames from tracebacks that include this module.

    :param mode: optional, the traceback mode.
    """
    hide_module_tracebacks(globals(), mode)


# hide by default.
hide_tracebacks(True)


def assert_tensor_views(*views: torch.Tensor) -> None:
    """
    Assert that each tensor is a view of the same storage..

    :param views: a series of child Tensors which must all be views of source.
    """
    if views:
        reference = views[0]
        views = views[1:]

    for t in views:
        eggs.assert_match(
            t.storage().data_ptr(),  # type: ignore
            reference.storage().data_ptr(),  # type: ignore
        )


def assert_tensor_storage_differs(
    tensor: torch.Tensor, reference: torch.Tensor
) -> None:
    """
    Assert that two tensors are not views of each other, and have different storage.

    :param tensor: the tensor.
    :param reference: the reference tensor.
    """
    eggs.assert_match(
        tensor.storage().data_ptr(),  # type: ignore
        hamcrest.not_(reference.storage().data_ptr()),  # type: ignore
    )


class TensorStructureMatcher(BaseMatcher):
    """
    PyHamcrest matcher for comparing the structure of a tensor to an exemplar.

    Matches:
      - device
      - size
      - dtype
      - layout
    """

    expected: torch.Tensor

    def __init__(self, expected: TensorConvertable) -> None:
        self.expected = torch.as_tensor(expected)

    @overrides
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

    @overrides
    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.expected)


def matches_tensor_structure(
    expected: TensorConvertable,
) -> TensorStructureMatcher:
    """
    Construct a matcher for comparing the structure of a tensor to an exemplar.

    Matches on:
      - device
      - size
      - dtype
      - layout

    :return: a matcher.
    """
    return TensorStructureMatcher(expected)


def assert_tensor_structure(
    actual: torch.Tensor,
    expected: TensorConvertable,
) -> None:
    """
    Assert that the `actual` matches the structure (not data) of the `expected`.

    :param actual: a tensor.
    :param expected: an expected structure.
    """
    hamcrest.assert_that(
        actual,
        matches_tensor_structure(expected),
    )


class TensorMatcher(TensorStructureMatcher):
    """
    PyHamcrest matcher for comparing the structure and data a tensor to an exemplar.

    Matches:
      - device
      - size
      - dtype
      - layout
    """

    close: bool = False
    "Should <close> values be considered identical?"

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

    @overrides
    def _matches(self, item: typing.Any) -> bool:
        if not super()._matches(item):
            return False

        if self.close:
            try:
                torch.testing.assert_close(
                    item,
                    self.expected,
                    equal_nan=True,
                )
                return True
            except AssertionError:
                return False

        else:
            if self.expected.is_sparse:
                eggs.assert_true(item.is_sparse)

                # TODO: it may be necessary to sort the indices and values.
                if not item.is_coalesced():
                    item = item.coalesce()

                assert_tensor_equals(
                    item.indices(),
                    self.expected.indices(),
                )
                assert_tensor_equals(
                    item.values(),
                    self.expected.values(),
                )
                return True
            else:
                # torch.equal(item, self.expected) does not support nan.
                try:
                    torch.testing.assert_close(
                        item,
                        self.expected,
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                except AssertionError:
                    return False
                return True

    @overrides
    def describe_to(self, description: Description) -> None:
        description.append_text("\n")
        description.append_description_of(self.expected)

    @overrides
    def describe_match(self, item: typing.Any, match_description: Description) -> None:
        match_description.append_text("was \n")
        match_description.append_description_of(item)

    @overrides
    def describe_mismatch(
        self, item: typing.Any, mismatch_description: Description
    ) -> None:
        torch.set_printoptions(
            precision=10,
        )
        mismatch_description.append_text("was \n")
        mismatch_description.append_description_of(item)


def matches_tensor(
    expected: TensorConvertable,
    close: bool = False,
) -> TensorMatcher:
    """
    Returns a matcher for structure and value of a Tensor.

    :param expected: the expected Tensor.
    :param close: should *close* values be acceptable?
    """
    return TensorMatcher(expected, close=close)


def assert_tensor_equals(
    actual: torch.Tensor,
    expected: TensorConvertable,
    *,
    close: bool = False,
    view_of: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Assert that the `actual` tensor equals the `expected` tensor.

    :param actual: the actual tensor.
    :param expected: the value (to coerce to a Tensor) to compare to.
    :param close: should *close* values match?
    :param view_of: if present, also check that actual is a view of the reference Tensor.
    :returns: the `actual` value.
    """
    hamcrest.assert_that(
        actual,
        matches_tensor(
            expected,
            close=close,
        ),
    )
    if view_of is not None:
        assert_tensor_views(view_of, actual)

    return actual


def match_tensor_sequence(
    *expected: TensorConvertable,
) -> Matcher:
    """
    Returns a matcher which expects a sequence that matches the tensors.
    :param expected: the expected tensors.
    """
    return hamcrest.contains_exactly(*[matches_tensor(e) for e in expected])


def assert_tensor_sequence_equals(
    actual: typing.Sequence[torch.Tensor],
    *expected: TensorConvertable,
    view_of: typing.Optional[torch.Tensor] = None,
) -> typing.Sequence[torch.Tensor]:
    """
    Assert that the `actual` is a sequence that equals the given `expected` tensor values.

    :param actual: the `actual` to test.
    :param expected: the expected values.
    :return: the `actual`
    """
    hamcrest.assert_that(
        actual,
        match_tensor_sequence(*expected),
    )
    if view_of is not None:
        assert_tensor_views(view_of, *actual)
    return actual


@contextlib.contextmanager
def reset_generator_seed(seed: int = 3 * 17 * 53 + 1) -> typing.Iterator:
    """
    Context manager which resets the `torch.manual_seed()` seed on entry.

    :param seed: optional seed.
    """
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
    close: bool = False,
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
