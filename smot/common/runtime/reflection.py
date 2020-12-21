import inspect
import os.path
import sys
from types import ModuleType
from typing import Optional

import IPython

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
    return os.path.dirname(module_directory(smot))


def this_module() -> ModuleType:
    """
    Return the calling module.

    :return: the module.
    """
    return calling_module()


def calling_module(
    *,
    module: Optional[ModuleType] = None,
    stack_depth: int = 1,
) -> ModuleType:
    """
    Extract the calling module from the frame at depth ``stack_depth``.

    stack_depth is measured relative to the caller, so ``stack_depth == 0`` is the caller.

    :param module: if present, the module to use.
    :param stack_depth: the stack depth, relative to the caller.
    :return: the module.
    :raises ValueError: if stack_depth is too deep or module not found.
    """
    if module is not None:
        return module

    frame = sys._getframe(stack_depth + 1)

    try:
        return sys.modules[frame.f_globals["__name__"]]

    except KeyError:
        raise ValueError("No Calling Module")


def module_directory(module: ModuleType) -> str:
    """
    Return a module's directory.

    :param module: the module.
    :return: the directory path.
    :raises ValueError: if stack_depth is too deep or module not found.
    """
    return str(os.path.dirname(module.__file__))


def module_name_as_relative_path(module: ModuleType) -> str:
    """
    Return the module's name as a relative path.

    Special cases

    Example module "foo.bar.baz" => "foo/bar/baz"

    Note, this is not a path to a file (no ".py") or directory.

    :param module: the module.
    :return: the path.
    """
    name = module.__name__

    return name.replace(".", "/")
