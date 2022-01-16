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


class TensorStructureMatcher(BaseMatcher):
    expected: torch.Tensor

    def __init__(self, expected):
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
    expected: typing.Union[
        torch.Tensor,
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> TensorStructureMatcher:
    return TensorStructureMatcher(expected)


def assert_tensor_structure(
    actual: torch.Tensor,
    expected: typing.Union[
        torch.Tensor,
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
):
    hamcrest.assert_that(
        actual,
        expect_tensor_structure(expected),
    )


class TensorMatcher(TensorStructureMatcher):
    close: bool = False

    def __init__(
        self,
        expected,
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
        description.append_description_of(self.expected)


def expect_tensor(
    expected: typing.Union[
        torch.Tensor,
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> TensorMatcher:
    return TensorMatcher(expected, close=False)


def expect_tensor_seq(
    *expected: typing.Union[
        torch.Tensor,
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> Matcher:
    return hamcrest.contains_exactly(*[expect_tensor(e) for e in expected])


def assert_tensor(
    actual: torch.Tensor,
    expected: typing.Union[
        torch.Tensor,
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
):
    hamcrest.assert_that(
        actual,
        expect_tensor(expected),
    )


def assert_tensor_seq(
    actual: typing.Sequence[torch.Tensor],
    *expected: typing.Union[
        torch.Tensor,
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
):
    hamcrest.assert_that(
        actual,
        expect_tensor_seq(*expected),
    )


def expect_tensor_close(
    expected: typing.Union[
        torch.Tensor,
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> TensorMatcher:
    return TensorMatcher(expected, close=True)


def assert_tensor_close(
    actual: torch.Tensor,
    expected: typing.Union[
        torch.Tensor,
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
):
    hamcrest.assert_that(
        actual,
        expect_tensor_close(expected),
    )
