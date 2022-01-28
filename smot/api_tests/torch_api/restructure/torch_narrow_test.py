import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.narrow",
    ref="https://pytorch.org/docs/stable/generated/torch.narrow.html",
)
class NarrowTest(unittest.TestCase):
    def test_narrow(self) -> None:
        source = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )

        view = torch.narrow(source, 0, 1, 2)

        torch_eggs.assert_tensor_views(source, view)

        torch_eggs.assert_tensor_equals(
            view,
            [
                [4, 5, 6],
                [7, 8, 9],
            ],
        )

        view = torch.narrow(source, 1, 0, 1)

        torch_eggs.assert_tensor_views(source, view)

        torch_eggs.assert_tensor_equals(
            view,
            [
                [1],
                [4],
                [7],
            ],
        )
