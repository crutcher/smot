from dataclasses import dataclass
import os
import sys
from types import ModuleType
from typing import Any, Callable, List, Optional, TypeVar

import smot
from smot.common.runtime import reflection

T = TypeVar("T")


@dataclass
class Location:
    rpath: str
    line: int


@dataclass
class Link:
    target: str
    location: Location
    ref: str
    doc: Optional[str] = None


LINKS = []


def api_link(
    target: Any,
    *,
    ref: str,
    doc: Optional[str] = None,
) -> Callable[[T], T]:
    def f(obj: T) -> T:
        frame = sys._getframe(1)
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        LINKS.append(
            Link(
                target=target,
                location=Location(
                    rpath=reflection.root_relative_path(filename),
                    line=lineno,
                ),
                ref=ref,
                doc=doc,
            ),
        )
        return obj

    return f


def collect_all_python_module_files(relative: bool = True) -> List[str]:
    results = []

    prefix = reflection.repository_source_root() + "/"
    root = reflection.module_directory(smot)

    for dirpath, dirnames, filenames in os.walk(root):
        prune = []
        for d in dirnames:
            if d.startswith(".") or d.startswith("_"):
                prune.append(d)

        for d in prune:
            dirnames.remove(d)

        for f in filenames:
            fpath = os.path.join(dirpath, f)
            if relative:
                fpath = fpath.removeprefix(prefix)

            if f.endswith(".py"):
                results.append(fpath)

    return results


def collect_all_python_module_names() -> List[str]:
    return [
        fpath.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        for fpath in collect_all_python_module_files()
    ]


def load_all_modules() -> List[ModuleType]:
    return [
        __import__(name)
        for name in collect_all_python_module_names()
        if name not in sys.modules
    ]


if __name__ == "__main__":
    from smot.api_tests import doc_links

    doc_links.load_all_modules()
    print(doc_links.LINKS)
