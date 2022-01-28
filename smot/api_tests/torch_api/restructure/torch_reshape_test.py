import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.reshape",
    ref="https://pytorch.org/docs/stable/generated/torch.reshape.html",
)
class ReshapeTest(unittest.TestCase):
    def test_reshape_view(self) -> None:
        source = torch.arange(4)

        # Contiguous reshapes produce views
        eggs.assert_true(source.is_contiguous())

        torch_eggs.assert_tensor_equals(
            torch.reshape(source, (-1,)),
            expected=[0, 1, 2, 3],
            view_of=source,
        )

        torch_eggs.assert_tensor_equals(
            torch.reshape(source, (2, 2)),
            expected=[[0, 1], [2, 3]],
            view_of=source,
        )

    def test_reshape_copy(self) -> None:
        source = torch.arange(9).reshape(3, 3).narrow(1, 0, 2)

        # Contiguous reshapes produce copies
        eggs.assert_false(source.is_contiguous())

        torch_eggs.assert_tensor_equals(
            source,
            [
                [0, 1],
                [3, 4],
                [6, 7],
            ],
        )

        torch_eggs.assert_tensor_storage_differs(
            torch.reshape(source, (-1,)),
            source,
        )

        torch_eggs.assert_tensor_equals(
            torch.reshape(source, (-1,)),
            [0, 1, 3, 4, 6, 7],
        )
