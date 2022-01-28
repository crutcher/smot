import contextlib
from dataclasses import dataclass
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Type,
)
import warnings

import hamcrest
from hamcrest.core.assert_that import _assert_bool, _assert_match
from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.description import Description
from hamcrest.core.matcher import Matcher


def hide_module_tracebacks(module: typing.Any, mode: bool = True) -> None:
    # unittest integration; hide these frames from tracebacks
    module["__unittest"] = mode
    # py.test integration; hide these frames from tracebacks
    module["__tracebackhide__"] = mode


def hide_tracebacks(mode: bool = True) -> None:
    """
    Hint that some unittest stacks (unittest, pytest) should remove
    frames from tracebacks that include this module.

    :param mode: optional, the traceback mode.
    """
    hide_module_tracebacks(globals(), mode)


hide_tracebacks(True)

hide_module_tracebacks(hamcrest.core.base_matcher.__dict__)


@dataclass
class WhenCalledMatcher(BaseMatcher[Callable[..., Any]]):
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    method: Optional[str] = None

    def __init__(
        self,
        args: Sequence[Any],
        kwargs: Dict[str, Any],
        matcher: Matcher,
        method: Optional[str] = None,
    ) -> None:
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.matcher = matcher
        self.method = method

    def _matches(self, item: Callable) -> bool:
        return self.matcher.matches(self._call_item(item))

    def _call_item(self, item: Callable) -> Any:
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

    def describe_mismatch(
        self, item: Callable, mismatch_description: Description
    ) -> None:
        val = self._call_item(item)
        mismatch_description.append_text("was =>").append_description_of(val)


@dataclass
class WhenCalledBuilder:
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    method: Optional[str] = None

    def matches(self, matcher: Any) -> WhenCalledMatcher:
        return WhenCalledMatcher(
            args=self.args,
            kwargs=self.kwargs,
            matcher=_as_matcher(matcher),
            method=self.method,
        )


def when_called(*args: Any, **kwargs: Any) -> WhenCalledBuilder:
    return WhenCalledBuilder(args, kwargs)


def calling_method(method: str, *args: Any, **kwargs: Any) -> WhenCalledBuilder:
    return WhenCalledBuilder(args, kwargs, method=method)


def _as_matcher(matcher: Any) -> Matcher:
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


def assert_close_to(
    actual: Any,
    expected: Any,
    delta: Optional[SupportsFloat] = None,
    *,
    rtol: SupportsFloat = 1e-05,
    atol: SupportsFloat = 1e-08,
) -> None:
    """
    Asserts that two values are close to each other.

    :param actual: the actual value.
    :param expected: the expected value.
    :param delta: (optional) the tolerance.
    :param rtol: if delta is None, the relative tolerance.
    :param atol: if delta is None, the absolute tolerance.
    :return:
    """
    if delta is None:
        # numpy.isclose() pattern:
        delta = atol + rtol * abs(expected)
    assert_match(
        actual,
        hamcrest.close_to(expected, delta),
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
def ignore_warnings() -> Generator[None, None, None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
