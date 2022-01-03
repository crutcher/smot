import os
from typing import Any, Optional, Type, TypeVar

T = TypeVar("T")

# unittest integration; hide these frames from tracebacks
__unittest = True
# py.test integration; hide these frames from tracebacks
__tracebackhide__ = True


class Expect:
    """
    Contract Programming Expectations.
    """

    @staticmethod
    def not_none(
        actual: Optional[T],
        msg: str = "Value is not None",
        cls: Type[Exception] = AssertionError,
        **kwargs: Any,
    ) -> T:
        if actual is None:
            raise cls(
                msg
                % dict(
                    **kwargs,
                )
            )
        return actual

    @staticmethod
    def is_truthy(
        actual: Any,
        msg: str = "Value is not truthy: %(actual)s",
        cls: Type[Exception] = AssertionError,
        **kwargs: Any,
    ) -> None:
        """
        Expect that a value is is_truthy.

        :param actual: the value to test.
        :param msg: the message template.
        :param cls: the exception class.
        :param kwargs: any additional keyword args for the message.
        """
        if not actual:
            raise cls(
                msg
                % dict(
                    actual=actual,
                    **kwargs,
                )
            )

    @staticmethod
    def is_falsey(
        actual: Any,
        msg: str = "Value is not falsey: %(actual)s",
        cls: Type[Exception] = AssertionError,
        **kwargs: Any,
    ) -> None:
        """
        Expect that a value is falsey.

        :param actual: the value to test.
        :param msg: the message template.
        :param cls: the exception class.
        :param kwargs: any additional keyword args for the message.
        """
        if actual:
            raise cls(
                msg
                % dict(
                    actual=actual,
                    **kwargs,
                )
            )

    @staticmethod
    def is_eq(
        actual: Any,
        expected: Any,
        msg: str = "Value (%(actual)s) != (%(expected)s)",
        cls: Type[Exception] = AssertionError,
        **kwargs: Any,
    ) -> None:
        """
        Expect that a value is is_truthy.

        :param actual: the value to test.
        :param expected: the value to compare.
        :param msg: the message template.
        :param cls: the exception class.
        :param kwargs: any additional keyword args for the message.
        """
        if actual != expected:
            raise cls(
                msg
                % dict(
                    actual=actual,
                    expected=expected,
                    **kwargs,
                )
            )


class ExpectPath:
    """
    Path Contract Programming Expectations.
    """

    @staticmethod
    def is_file(
        path: str,
        msg: str = "Path (%(path)s) is not a file.",
        cls: Type[Exception] = AssertionError,
        **kwargs: Any,
    ) -> str:
        """
        Expect that a path exists as a file.

        :param path: the value to test.
        :param msg: the message template.
        :param cls: the exception class.
        :param kwargs: any additional keyword args for the message.
        :return: the path.
        """
        if not os.path.isfile(path):
            raise cls(
                msg
                % dict(
                    path=path,
                    **kwargs,
                )
            )

        return path
