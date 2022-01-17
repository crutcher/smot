import os.path

import pytest

from smot.common.runtime import reflection


def pytest_addoption(parser):
    parser.addoption(
        "--gen_api_ref",
        action="store",
        default=None,
        help="Dump the api reference links document",
    )


def pytest_collection_finish(session):
    if session.config.option.gen_api_ref is not None:
        seen = set()
        es = []

        for item in session.items:
            testcase = item.getparent(pytest.Class)
            if testcase in seen:
                continue
            seen.add(testcase)

            if hasattr(testcase, "_obj"):
                obj = testcase._obj
                if not hasattr(obj, "API_DOC"):
                    continue

                e = {}
                es.append(e)

                e["path"] = reflection.root_relative_path(str(item.fspath))
                e["dir"] = os.path.dirname(e["path"])
                e["api_doc"] = obj.API_DOC

                e["target"] = obj.TARGET
                e["target_name"] = obj.target_name()

        with open(session.config.option.gen_api_ref, "wt") as fh:
            es = sorted(es, key=lambda i: (i["dir"], i["target_name"]))
            for e in es:
                parts = [
                    f"### [{e['target_name']}]({e['path']})",
                    f"{e['api_doc']}",
                ]
                docs = " ".join(e["target"].__doc__.strip().splitlines()[0].split())
                parts.extend(["", "    " + docs])

                parts += ["", ""]

                print("\n".join(parts), file=fh)

        pytest.exit("Doc Gen Complete", 0)
