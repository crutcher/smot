import inspect
import io
import os
import pydoc
import sys
import textwrap
from types import ModuleType
from typing import Any, List

import smot
from smot.common.runtime import reflection
from smot.doc_link import link_annotations


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


def format_help(obj: Any) -> str:
    reformat = "no __doc__"

    if inspect.isclass(obj) and obj.__doc__:
        lines = obj.__doc__.splitlines()
        first = lines[0].strip()
        body = textwrap.dedent("\n".join(lines[1:]))
        reformat = "\n".join([first, body])

    else:
        render_lines = pydoc.render_doc(obj).splitlines()
        reformat = "\n".join(render_lines[3:])

    return reformat


def render_api_index(*, show_help: bool = False) -> str:
    buffer = io.StringIO()

    load_all_modules()
    link_dict = {link.target: link for link in link_annotations.API_LINKS}
    for link in dict(sorted(link_dict.items())).values():
        print(
            "\n".join(
                [
                    "",
                    f"#### {link.target}",
                    (
                        # GitHub will follow '#L<LINENO>' markdown links,
                        # IntelliJ will not, render both of them.
                        # https://youtrack.jetbrains.com/issue/IDEA-287198
                        f"  * Tests: [{link.location.rpath}]({link.location.rpath})"
                        + f" [L{link.location.line}]({link.location.rpath}#L{link.location.line})"
                    ),
                    f"  * Docs: [{link.ref}]({link.ref})",
                    "",
                ]
            ),
            file=buffer,
        )

        if link.aliases:
            print("Aliases:", file=buffer)
            for a in link.aliases:
                alias_href = "#" + a.replace(".", "").replace(" ", "-")
                print(f"    * [{a}]({alias_href})", file=buffer)

        if link.link_target:
            if show_help:
                print(file=buffer)
                h = format_help(link.link_target.object)
                print("\n".join([textwrap.indent(h, "    ")]), file=buffer)

    return buffer.getvalue()


def main(argv: List[str]) -> None:
    print(render_api_index())


if __name__ == "__main__":
    main(sys.argv)
