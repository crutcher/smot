import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type
import warnings

import hamcrest
from hamcrest.core.assert_that import _assert_bool, _assert_match
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher

# unittest integration; hide these frames from tracebacks

__unittest = True
# py.test integration; hide these frames from tracebacks
__tracebackhide__ = True

# Monkey patch BaseMatcher
hamcrest.core.base_matcher.__unittest = True
hamcrest.core.base_matcher.__tracebackhide__ = True


@dataclass
class WhenCalledMatcher(BaseMatcher):
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    method: Optional[str] = None

    def __init__(
        self,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
        matcher: Matcher,
        method: Optional[str] = None,
    ):
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.matcher = matcher
        self.method = method

    def _matches(self, item) -> bool:
        return self.matcher.matches(self._call_item(item))

    def _call_item(self, item) -> Any:
        if self.method is None:
            f = item
        else:
            f = getattr(item, self.method)

        return f(*self.args, **self.kwargs)

    def describe_to(self, description: Description) -> None:
        call_sig = tuple(
            [repr(a) for a in self.args]
            + [f"{k}={repr(v)}" for k, v in self.kwargs.items()]
        )

        if self.method is None:
            f = "\\"
        else:
            f = "." + self.method

        description.append_text(f"{f}{call_sig}=>")

        description.append_description_of(self.matcher)

    def describe_mismatch(self, item: Any, mismatch_description: Description) -> None:
        val = self._call_item(item)
        mismatch_description.append_text("was =>").append_description_of(val)


@dataclass
class WhenCalledBuilder:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    method: Optional[str] = None

    def matches(self, matcher) -> WhenCalledMatcher:
        return WhenCalledMatcher(
            args=self.args,
            kwargs=self.kwargs,
            matcher=_as_matcher(matcher),
            method=self.method,
        )


def when_called(*args, **kwargs) -> WhenCalledBuilder:
    return WhenCalledBuilder(args, kwargs)


def calling_method(method, *args, **kwargs) -> WhenCalledBuilder:
    return WhenCalledBuilder(args, kwargs, method=method)


def _as_matcher(matcher) -> Matcher:
    if not isinstance(matcher, Matcher):
        matcher = hamcrest.is_(matcher)

    return matcher


def assert_match(actual: Any, matcher: Any, reason: str = "") -> None:
    """
    Asserts that the actual value matches the matcher.

    Similar to hamcrest.assert_that(), but if the matcher is not a Matcher,
    will fallback to ``hamcrest.is_(matcher)`` rather than boolean matching.

    :param actual: the value to match.
    :param matcher: a matcher, or a value that will be converted to an ``is_`` matcher.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_match(
        actual=actual,
        matcher=_as_matcher(matcher),
        reason=reason,
    )


def assert_true(actual: Any, reason: str = "") -> None:
    """
    Asserts that the actual value is truthy.

    :param actual: the value to match.
    :param reason: Optional explanation to include in failure description.
    """
    _assert_bool(actual, reason=reason)


def assert_false(actual: Any, reason: str = "") -> None:
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


@contextlib.contextmanager
def ignore_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
