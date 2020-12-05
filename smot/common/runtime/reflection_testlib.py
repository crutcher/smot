from typing import Any, Callable


def apply(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
  """
  Utility mechanism to add an indirection module for testing stack reflection.

  :param func: the func to call.
  :param args: the positional args.
  :param kwargs: the keyword args.
  :return: the return value.
  """
  return func(*args, **kwargs)
