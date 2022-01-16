import numbers
import typing

import hamcrest
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher
import nptyping
import numpy as np

from smot.testlib import eggs

# unittest integration; hide these frames from tracebacks
__unittest = True
# py.test integration; hide these frames from tracebacks
__tracebackhide__ = True


class NDArrayStructureMatcher(BaseMatcher):
    expected: np.ndarray

    def __init__(self, expected):
        self.expected = np.asarray(expected)

    def _matches(self, item) -> bool:
        # Todo: structural miss-match that still shows expected ndarray.

        try:
            eggs.assert_match(
                item.shape,
                self.expected.shape,
            )
            eggs.assert_match(
                item.dtype,
                self.expected.dtype,
            )
            return True
        except AssertionError:
            return False

    def describe_to(self, description: Description) -> None:
        description.append_description_of(self.expected)


def expect_ndarray_structure(
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> NDArrayStructureMatcher:
    return NDArrayStructureMatcher(expected)


def assert_ndarray_structure(
    actual: np.ndarray,
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
):
    hamcrest.assert_that(
        actual,
        expect_ndarray_structure(expected),
    )


class NDArrayMatcher(NDArrayStructureMatcher):
    close: bool = False

    def __init__(
        self,
        expected,
        *,
        close: bool = False,
    ):
        super().__init__(expected=expected)
        self.close = close

    def _matches(self, item) -> bool:
        if not super()._matches(item):
            return False

        if self.close:
            np.testing.assert_allclose(
                item,
                self.expected,
                equal_nan=True,
            )

        else:
            np.testing.assert_equal(item, self.expected)

        return True

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


def expect_ndarray(
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> NDArrayMatcher:
    return NDArrayMatcher(expected, close=False)


def expect_ndarray_seq(
    *expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> Matcher:
    return hamcrest.contains_exactly(*[expect_ndarray(e) for e in expected])


def assert_ndarray(
    actual: np.ndarray,
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
):
    hamcrest.assert_that(
        actual,
        expect_ndarray(expected),
    )


def assert_ndarray_seq(
    actual: typing.Sequence[np.ndarray],
    *expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
):
    hamcrest.assert_that(
        actual,
        expect_ndarray_seq(*expected),
    )


def expect_ndarray_close(
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
) -> NDArrayMatcher:
    return NDArrayMatcher(expected, close=True)


def assert_ndarray_close(
    actual: np.ndarray,
    expected: typing.Union[
        numbers.Number,
        typing.Sequence,
        nptyping.NDArray,
    ],
):
    hamcrest.assert_that(
        actual,
        expect_ndarray_close(expected),
    )
