from dataclasses import dataclass
import sys
from types import ModuleType
from typing import Any, Callable, List, Optional, TypeVar, Union

from smot.common.runtime import reflection

# unittest integration; hide these frames from tracebacks
__unittest = True
# py.test integration; hide these frames from tracebacks
__tracebackhide__ = True

T = TypeVar("T")


@dataclass
class Location:
    rpath: str
    line: int


@dataclass
class LinkTarget:
    target: str
    module: ModuleType
    module_name: str
    object_name: str
    object: Any


@dataclass
class Link:
    target: str
    aliases: List[str]
    location: Location
    ref: str
    doc: Optional[str] = None
    link_target: Optional[LinkTarget] = None


API_LINKS: List[Link] = []
VERIFY: bool = True


def _find_target(qual_name: str) -> LinkTarget:
    parts = qual_name.split(".")

    mod_parts: List[str] = []
    mod_name = ".".join(mod_parts)
    probe_name = ""

    assert parts, "Empty target name."

    for p in parts:
        # look for the module
        probe_name = ".".join(mod_parts + [p])
        try:
            mod = __import__(
                probe_name,
                fromlist=[None] if "." in probe_name else [],  # type: ignore
            )
        except ImportError:
            break

        mod_parts.append(p)
        mod_name = probe_name
    else:
        raise ImportError(f'Module "{probe_name}" not found in "{mod_name}"')

    # mod is module
    # mod_name is the name
    parts = parts[len(mod_parts) :]

    tree = [mod]
    obj_parts: List[str] = []
    for p in parts:
        try:
            tree.append(getattr(tree[-1], p))
            obj_parts.append(p)
        except AttributeError:
            parent_name = ".".join(obj_parts)
            raise AttributeError(f'"{parent_name}" has no attribute "{p}"')

    return LinkTarget(
        target=qual_name,
        module=mod,
        module_name=mod_name,
        object=tree[-1],
        object_name=".".join(parts),
    )


def _verify_target(target: str) -> LinkTarget:
    try:
        return _find_target(target)
    except AttributeError as e:
        raise AssertionError(f"@api_link() could not resolve target: {target}")


def api_link(
    target: Any,
    *,
    ref: str,
    note: Optional[str] = None,
    alias: Optional[Union[str, List[str]]] = None,
) -> Callable[[T], T]:
    if not alias:
        alias = []
    if isinstance(alias, str):
        alias = [alias]
    aliases: List[str] = list(alias)

    def hook(obj: T) -> T:
        frame = sys._getframe(1)
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno

        if VERIFY:
            link_target = _verify_target(target)
            for alias in aliases:
                _verify_target(alias)
        else:
            link_target = None

        API_LINKS.append(
            Link(
                target=target,
                aliases=aliases,
                location=Location(
                    rpath=reflection.root_relative_path(filename),
                    line=lineno,
                ),
                ref=ref,
                doc=note,
                link_target=link_target,
            ),
        )
        return obj

    return hook


def WEIRD_BUG(
    target: Any,
    note: str,
) -> None:
    if VERIFY:
        link_target = _verify_target(target)

    # TODO: link this in docs.


def WEIRD_API(
    target: Any,
    note: str,
) -> None:
    if VERIFY:
        link_target = _verify_target(target)

    # TODO: link this in docs.
