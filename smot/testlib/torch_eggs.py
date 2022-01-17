import numbers
import typing

import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
import nptyping
import torch

from smot.testlib import eggs

# unittest integration; hide these frames from tracebacks
__unittest = True
# py.test integration; hide these frames from tracebacks
__tracebackhide__ = True


def assert_views(source: torch.Tensor, *tensor: torch.Tensor):
    for t in tensor:
        eggs.assert_match(
            t.storage().data_ptr(),  # type: ignore
            source.storage().data_ptr(),  # type: ignore
        )


def assert_not_view(tensor: torch.Tensor, source: torch.Tensor):
    eggs.assert_match(
        tensor.storage().data_ptr(),  # type: ignore
        hamcrest.not_(source.storage().data_ptr()),  # type: ignore
    )


ExpectedType = typing.Union[
    torch.Tensor,
    numbers.Number,
    typing.Sequence,
    nptyping.NDArray,
]


class TensorStructureMatcher(BaseMatcher):
    expected: torch.Tensor

    def __init__(self, expected: ExpectedType):
        self.expected = torch.as_tensor(expected)

    def _matches(self, item) -> bool:
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
    expected: ExpectedType,
) -> TensorStructureMatcher:
    return TensorStructureMatcher(expected)


def assert_tensor_structure(
    actual: torch.Tensor,
    expected: ExpectedType,
):
    hamcrest.assert_that(
        actual,
        expect_tensor_structure(expected),
    )


class TensorMatcher(TensorStructureMatcher):
    close: bool = False

    def __init__(
        self,
        expected: ExpectedType,
        *,
        close: bool = False,
    ):
        super().__init__(expected=expected)
        self.close = close

        if self.expected.is_sparse and not self.expected.is_coalesced():
            self.expected = self.expected.coalesce()

    def _matches(self, item) -> bool:
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
    expected: ExpectedType,
) -> TensorMatcher:
    return TensorMatcher(expected, close=False)


def expect_tensor_seq(
    *expected: ExpectedType,
) -> Matcher:
    return hamcrest.contains_exactly(*[expect_tensor(e) for e in expected])


def assert_tensor(
    actual: torch.Tensor,
    expected: ExpectedType,
):
    hamcrest.assert_that(
        actual,
        expect_tensor(expected),
    )


def assert_view_tensor(
    actual: torch.Tensor,
    source: torch.Tensor,
    expected: ExpectedType,
):
    assert_views(source, actual)
    assert_tensor(actual, expected)


def assert_tensor_seq(
    actual: typing.Sequence[torch.Tensor],
    *expected: ExpectedType,
):
    hamcrest.assert_that(
        actual,
        expect_tensor_seq(*expected),
    )


def assert_view_tensor_seq(
    actual: typing.Sequence[torch.Tensor],
    source: torch.Tensor,
    *expected: ExpectedType,
):
    assert_views(source, *actual)
    hamcrest.assert_that(
        actual,
        expect_tensor_seq(*expected),
    )


def expect_tensor_close(
    expected: ExpectedType,
) -> TensorMatcher:
    return TensorMatcher(expected, close=True)


def assert_tensor_close(
    actual: torch.Tensor,
    expected: ExpectedType,
):
    hamcrest.assert_that(
        actual,
        expect_tensor_close(expected),
    )
