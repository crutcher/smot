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
    path: str = None,
    *,
    module: Optional[ModuleType] = None,
    stack_depth: int = 1,
    create: bool = True,
) -> str:
    module = reflection.calling_module(module=module, stack_depth=stack_depth)

    p = os.path.join(
        build_root(),
        "out",
        reflection.module_name_as_relative_path(module),
    )
    if create:
        os.makedirs(p, exist_ok=True)

    if path:
        p = os.path.join(p, path)

    return p
