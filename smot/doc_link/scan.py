import os
import pydoc
import sys
from types import ModuleType
from typing import List

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


def gen_index() -> None:
    load_all_modules()
    link_dict = {link.target: link for link in link_annotations.API_LINKS}
    for link in dict(sorted(link_dict.items())).values():
        print(
            "\n".join(
                [
                    "",
                    f"#### {link.target}",
                    f"[{link.location.rpath}]({link.location.rpath}#L{link.location.line})",
                    f"See: [{link.ref}]({link.ref})",
                    "",
                ]
            )
        )

        if link.aliases:
            print("Aliases:")
            for a in link.aliases:
                print(f"    * [{a}](#a)")

        if link.link_target:
            render_lines = pydoc.render_doc(
                link.link_target.object,
                title="Help on %s:",
            ).splitlines()
            print(render_lines[0])
            print()
            print("\n".join(render_lines[3:]))


def main(argv: List[str]) -> None:
    gen_index()


if __name__ == "__main__":
    main(sys.argv)
