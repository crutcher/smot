from typing import Any, Type, TypeVar

T = TypeVar('T')

# unittest integration; hide these frames from tracebacks
__unittest = True
# py.test integration; hide these frames from tracebacks
__tracebackhide__ = True


class Expect:
  """
  Contract Programming Expectations.
  """

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
      raise cls(msg % dict(
        actual=actual,
        **kwargs,
      ))

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
      raise cls(msg % dict(
        actual=actual,
        **kwargs,
      ))

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
      raise cls(msg % dict(
        actual=actual,
        expected=expected,
        **kwargs,
      ))
