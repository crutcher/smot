from dataclasses import dataclass
from typing import Dict
import unittest

import hamcrest
import testfixtures
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@dataclass
class Foo:
    tensor: torch.Tensor
    map: Dict[str, torch.Tensor]


@api_link(
    target="torch.save",
    ref="https://pytorch.org/docs/stable/generated/torch.save.html",
)
@api_link(
    target="torch.load",
    ref="https://pytorch.org/docs/stable/generated/torch.load.html",
)
class SaveAndLoadTest(unittest.TestCase):
    @testfixtures.tempdir()
    def test_tensor_path(self, tempdir: testfixtures.TempDirectory) -> None:
        source = torch.arange(9).reshape(3, 3)

        path = tempdir.getpath("tensor.pt")

        torch.save(source, path)

        target = torch.load(path)

        torch_eggs.assert_tensor_equals(
            target,
            source,
        )

    @testfixtures.tempdir()
    def test_tensor_filestream(self, tempdir: testfixtures.TempDirectory) -> None:
        source = torch.arange(9).reshape(3, 3)

        path = tempdir.getpath("tensor.pt")
        with open(path, "wb") as f:
            torch.save(source, f)

        with open(path, "rb") as f:
            target = torch.load(f)

        torch_eggs.assert_tensor_equals(
            target,
            source,
        )

    @testfixtures.tempdir()
    def test_classes(self, tempdir: testfixtures.TempDirectory) -> None:
        source = Foo(
            tensor=torch.ones(2, 3),
            map={"abc": torch.arange(4)},
        )

        path = tempdir.getpath("dat.pt")
        torch.save(source, path)
        target = torch.load(path)

        eggs.assert_match(
            target,
            hamcrest.all_of(
                hamcrest.instance_of(Foo),
                hamcrest.has_properties(
                    tensor=torch_eggs.matches_tensor(torch.ones(2, 3)),
                    map=hamcrest.has_entries(
                        abc=torch_eggs.matches_tensor(torch.arange(4)),
                    ),
                ),
            ),
        )

    @testfixtures.tempdir()
    def test_storage_sharing(self, tempdir: testfixtures.TempDirectory) -> None:
        source = torch.arange(9).reshape(3, 3)
        splits = source.vsplit(3)
        torch_eggs.assert_tensor_views(source, *splits)

        d = {
            "source": source,
            "splits": splits,
        }

        path = tempdir.getpath("d.pt")

        torch.save(d, path)

        target = torch.load(path)

        print(target)

        eggs.assert_match(
            target,
            hamcrest.has_entries(
                source=torch_eggs.matches_tensor(source),
                splits=torch_eggs.match_tensor_sequence(*splits),
            ),
        )
        # Note, the loaded views preserve storage sharing!
        torch_eggs.assert_tensor_views(
            target["source"],
            *target["splits"],
        )
