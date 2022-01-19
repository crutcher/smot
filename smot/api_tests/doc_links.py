from types import ModuleType
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def api_link(
    module: ModuleType,
    target: Any,
    ref: str,
) -> Callable[[T], T]:
    def f(obj: T) -> T:
        return obj

    return f
