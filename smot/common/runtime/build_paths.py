import os
from types import ModuleType
from typing import Optional

from smot.common.runtime import reflection


def build_root() -> str:
    """
    The configured build root root for cached data.
    """

    return os.path.join(
        reflection.repository_source_root(),
        "build",
    )


def data_root() -> str:
    """
    The configured source root for cached data.
    """
    return os.path.join(
        build_root(),
        "data_cache",
    )


def module_output_path(
    filename: str = None,
    *,
    module: Optional[ModuleType] = None,
    stack_depth: int = 1,
    create: bool = True,
) -> str:
    """
    Path in the build tree for module output.

    :param filename: the filename.
    :param module: the (optional) module to base the path on.
    :param stack_depth: the stack depth to find the calling module.
    :param create: create the enclosing directory.
    :return: the resolved path.
    """
    module = reflection.calling_module(module=module, stack_depth=stack_depth)

    p = os.path.join(
        build_root(),
        "out",
        reflection.module_name_as_relative_path(module),
    )
    if create:
        os.makedirs(p, exist_ok=True)

    if filename:
        p = os.path.join(p, filename)

    return p
