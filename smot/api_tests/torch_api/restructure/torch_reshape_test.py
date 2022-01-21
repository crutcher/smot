import unittest

import torch

from smot.api_tests.doc_links import api_link
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

        torch_eggs.assert_view_tensor(
            torch.reshape(source, (-1,)),
            source,
            [0, 1, 2, 3],
        )

        torch_eggs.assert_view_tensor(
            torch.reshape(source, (2, 2)),
            source,
            [[0, 1], [2, 3]],
        )

    def test_reshape_copy(self) -> None:
        source = torch.arange(9).reshape(3, 3).narrow(1, 0, 2)

        # Contiguous reshapes produce copies
        eggs.assert_false(source.is_contiguous())

        torch_eggs.assert_tensor(
            source,
            [
                [0, 1],
                [3, 4],
                [6, 7],
            ],
        )

        torch_eggs.assert_not_view(
            torch.reshape(source, (-1,)),
            source,
        )

        torch_eggs.assert_tensor(
            torch.reshape(source, (-1,)),
            [0, 1, 3, 4, 6, 7],
        )
