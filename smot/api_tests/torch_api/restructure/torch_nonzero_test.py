import unittest

import hamcrest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.nonzero",
    ref="https://pytorch.org/docs/stable/generated/torch.nonzero.html",
)
class NonzeroTest(unittest.TestCase):
    def test_1d(self) -> None:
        source = torch.tensor([1, 2, 3, 0, 4])
        torch_eggs.assert_tensor(
            torch.nonzero(source),
            [[0], [1], [2], [4]],
        )

        eggs.assert_match(
            torch.nonzero(source, as_tuple=True),
            hamcrest.contains_exactly(
                torch_eggs.expect_tensor([0, 1, 2, 4]),
            ),
        )

        torch_eggs.assert_tensor(
            source[torch.nonzero(source, as_tuple=True)],
            [1, 2, 3, 4],
        )

    def test_2d(self) -> None:
        source = torch.tensor(
            [
                [1, 2, 3, 0, 4],
                [0, 5, 0, 6, 7],
            ]
        )

        torch_eggs.assert_tensor(
            torch.nonzero(source),
            [[0, 0], [0, 1], [0, 2], [0, 4], [1, 1], [1, 3], [1, 4]],
        )

        eggs.assert_match(
            torch.nonzero(source, as_tuple=True),
            hamcrest.contains_exactly(
                torch_eggs.expect_tensor([0, 0, 0, 0, 1, 1, 1]),
                torch_eggs.expect_tensor([0, 1, 2, 4, 1, 3, 4]),
            ),
        )

        torch_eggs.assert_tensor(
            source[torch.nonzero(source, as_tuple=True)],
            [1, 2, 3, 4, 5, 6, 7],
        )
