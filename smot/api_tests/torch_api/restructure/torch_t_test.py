import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.t",
    ref="https://pytorch.org/docs/stable/generated/torch.t.html",
)
class TTest(unittest.TestCase):
    def test_t_scalar(self) -> None:
        # <= 2 dimensions ...
        source = torch.tensor(3)
        torch_eggs.assert_tensor_equals(torch.t(source), expected=3, view_of=source)

    def test_t_1d(self) -> None:
        # <= 2 dimensions ...
        source = torch.tensor([3, 2])
        torch_eggs.assert_tensor_equals(
            torch.t(source), expected=[3, 2], view_of=source
        )

    def test_t(self) -> None:
        source = torch.arange(6).reshape(2, 3)
        torch_eggs.assert_tensor_equals(
            source,
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
        )

        torch_eggs.assert_tensor_equals(
            torch.t(source),
            expected=[
                [0, 3],
                [1, 4],
                [2, 5],
            ],
            view_of=source,
        )

    def test_error(self) -> None:
        source = torch.arange(6).reshape(2, 1, 3)
        eggs.assert_raises(
            lambda: torch.t(source),
            RuntimeError,
            "expects a tensor with <= 2 dimensions",
        )
