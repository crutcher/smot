from typing import NoReturn, Type
import unittest

import hamcrest

from smot.testing import eggs


def throw(exception: Exception) -> NoReturn:
    """
    Throw an exception, useful inside a lambad.

    :param exception: the exception.
    """
    raise exception


class AssertMatchTest(unittest.TestCase):
    def test(self) -> None:
        eggs.assert_match("abc", "abc")

        hamcrest.assert_that(
            lambda: eggs.assert_match("abc", "xyz"),
            hamcrest.raises(
                AssertionError,
                "Expected: 'xyz'",
            ),
        )


class AssertTruthyTest(unittest.TestCase):
    def test(self) -> None:
        eggs.assert_truthy(True)
        eggs.assert_truthy("abc")
        eggs.assert_truthy(1)
        eggs.assert_truthy([1])

        hamcrest.assert_that(
            lambda: eggs.assert_truthy(False),
            hamcrest.raises(
                AssertionError,
            ),
        )

        hamcrest.assert_that(
            lambda: eggs.assert_truthy("", reason="meh"),
            hamcrest.raises(
                AssertionError,
                "meh",
            ),
        )


class AssertFalseyTest(unittest.TestCase):
    def test(self) -> None:
        eggs.assert_falsey(False)
        eggs.assert_falsey("")
        eggs.assert_falsey(0)
        eggs.assert_falsey([])

        hamcrest.assert_that(
            lambda: eggs.assert_falsey(True),
            hamcrest.raises(
                AssertionError,
            ),
        )

        hamcrest.assert_that(
            lambda: eggs.assert_falsey("abc", reason="meh"),
            hamcrest.raises(
                AssertionError,
                "meh",
            ),
        )


class AssertRaisesTest(unittest.TestCase):
    def test_simple(self) -> None:
        eggs.assert_raises(
            lambda: throw(ValueError("abc")),
            ValueError,
        )

        eggs.assert_raises(lambda: throw(ValueError("abc")), ValueError, "abc")

        # No exception.
        hamcrest.assert_that(
            lambda: eggs.assert_raises(
                lambda: (),
                ValueError,
            ),
            hamcrest.raises(
                AssertionError,
                "No exception raised",
            ),
        )

        # Wrong exception type.
        hamcrest.assert_that(
            lambda: eggs.assert_raises(
                lambda: throw(ValueError("abc 123")), IndexError, "abc [0-9]+"
            ),
            hamcrest.raises(
                AssertionError,
                "was raised instead",
            ),
        )

    def test_regex(self) -> None:
        eggs.assert_raises(
            lambda: throw(ValueError("abc 123")), ValueError, "abc [0-9]+"
        )

        hamcrest.assert_that(
            lambda: eggs.assert_raises(
                lambda: throw(ValueError("abc xyz")), ValueError, "abc [0-9]+"
            ),
            hamcrest.raises(
                AssertionError,
                "the expected pattern .* not found",
            ),
        )

    def test_matching(self) -> None:
        class ExampleException(ValueError):
            code: int

        e = ExampleException("abc 123")
        e.code = 123

        eggs.assert_raises(
            lambda: throw(e),
            ValueError,
            matching=hamcrest.has_properties(code=123),
        )

        hamcrest.assert_that(
            lambda: eggs.assert_raises(
                lambda: throw(e),
                ValueError,
                matching=hamcrest.has_properties(code=9),
            ),
            hamcrest.raises(
                AssertionError,
                "Correct assertion type .* but an object with .* not found",
            ),
        )
