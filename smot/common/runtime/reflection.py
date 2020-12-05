import inspect
import os.path
from types import ModuleType
from typing import Optional

import sys

import smot

# unittest integration; hide these frames from tracebacks
__unittest = True
# py.test integration; hide these frames from tracebacks
__tracebackhide__ = True


def repository_source_root() -> str:
  """
  Return the source root of the repository.

  :return: the source root.
  """
  return module_directory(smot)


def calling_module(stack_depth: int = 1) -> ModuleType:
  """
  Extract the calling module from the frame at depth ``stack_depth``.

  stack_depth is measured relative to the caller, so ``stack_depth == 0`` is the caller.

  :param stack_depth: the stack depth, relative to the caller.
  :return: the module.
  :raises ValueError: if stack_depth is too deep or module not found.
  """
  frame = sys._getframe(stack_depth + 1)

  if module := inspect.getmodule(frame):
    return module

  raise ValueError("No Calling Module")


def module_directory(
  module: Optional[ModuleType] = None,
  *,
  stack_depth: int = 1,
) -> str:
  """
  Return a module's directory.

  stack_depth is measured relative to the caller, so ``stack_depth == 0`` is the caller.

  :param module: if not-None, the module to use; ignore stack_depth.
  :param stack_depth: the stack depth, relative to the caller.
  :return: the directory path.
  :raises ValueError: if stack_depth is too deep or module not found.
  """
  if not module:
    module = calling_module(stack_depth + 1)

  return str(os.path.dirname(module.__file__))
