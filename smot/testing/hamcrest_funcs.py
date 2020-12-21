from typing import Any, Callable, Optional, Type

import hamcrest
from hamcrest.core.assert_that import _assert_bool, _assert_match
from hamcrest.core.matcher import Matcher

# unittest integration; hide these frames from tracebacks
__unittest = True
# py.test integration; hide these frames from tracebacks
__tracebackhide__ = True


def assert_match(actual: Any, matcher: Any, reason: str = "") -> None:
    """
    Asserts that the actual value matches the matcher.

    Similar to hamcrest.assert_that(), but if the matcher is not a Matcher,
    will fallback to ``hamcrest.is_(matcher)`` rather than boolean matching.

    :param actual: the value to match.
    :param matcher: a matcher, or a value that will be converted to an ``is_`` matcher.
    :param reason: Optional explanation to include in failure description.
    """
    if not isinstance(matcher, Matcher):
        matcher = hamcrest.is_(matcher)

    _assert_match(
        actual=actual,
        matcher=matcher,
        reason=reason,
    )


def assert_truthy(actual: Any, reason: str = "") -> None:
    """
    Asserts that the actual value is truthy.

    :param actual: the value to match.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_bool(actual, reason=reason)


def assert_falsey(actual: Any, reason: str = "") -> None:
    """
    Asserts that the actual value is falsey.

    :param actual: the value to match.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_bool(not actual, reason=reason)


def assert_raises(
    func: Callable[[], Any],
    exception: Type[Exception],
    pattern: Optional[str] = None,
    matching: Any = None,
) -> None:
    """
    Utility wrapper for ``hamcrest.assert_that(func, hamcrest.raises(...))``.

    :param func: the function to call.
    :param exception: the exception class to expect.
    :param pattern: an optional regex to match against the exception message.
    :param matching: optional Matchers to match against the exception.
    """

    hamcrest.assert_that(
        func,
        hamcrest.raises(
            exception=exception,
            pattern=pattern,
            matching=matching,
        ),
    )
