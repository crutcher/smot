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
  def truthy(
    test: bool,
    msg: str = "Value is not truthy.",
    cls: Type[Exception] = AssertionError,
    **kwargs: Any,
  ) -> None:
    """
    Expect that a value is truthy.

    :param test: the value to test.
    :param msg: the message template.
    :param cls: the exception class.
    :param kwargs: any additional keyword args for the message.
    """
    if not test:
      raise cls(msg % dict(kwargs))
