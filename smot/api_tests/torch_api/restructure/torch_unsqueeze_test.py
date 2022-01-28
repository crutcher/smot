import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.unsqueeze",
    ref="https://pytorch.org/docs/stable/generated/torch.unsqueeze.html",
)
class UnsqueezeTest(unittest.TestCase):
    def test_scalar(self) -> None:
        source = torch.tensor(3)

        torch_eggs.assert_tensor_equals(
            torch.unsqueeze(source, 0), expected=[3], view_of=source
        )

    def test_unsqueeze(self) -> None:
        source = torch.tensor([1, 2, 3, 4])

        torch_eggs.assert_tensor_equals(
            torch.unsqueeze(source, 0), expected=[[1, 2, 3, 4]], view_of=source
        )

        torch_eggs.assert_tensor_equals(
            torch.unsqueeze(source, 1),
            expected=[[1], [2], [3], [4]],
            view_of=source,
        )

        # negative dims

        torch_eggs.assert_tensor_equals(
            torch.unsqueeze(source, -1),
            expected=[[1], [2], [3], [4]],
            view_of=source,
        )

        torch_eggs.assert_tensor_equals(
            torch.unsqueeze(source, -2), expected=[[1, 2, 3, 4]], view_of=source
        )

    def test_error(self) -> None:
        source = torch.tensor([1, 2, 3, 4])

        eggs.assert_raises(
            lambda: torch.unsqueeze(source, 2), IndexError, "Dimension out of range"
        )

        eggs.assert_raises(
            lambda: torch.unsqueeze(source, -3), IndexError, "Dimension out of range"
        )
