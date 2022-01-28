import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.masked_select",
    ref="https://pytorch.org/docs/stable/generated/torch.masked_select.html",
)
class MaskedSelectTest(unittest.TestCase):
    def test_select(self) -> None:
        source = torch.arange(9).reshape(3, 3)

        torch_eggs.assert_tensor_equals(
            source,
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
        )

        torch_eggs.assert_tensor_equals(
            torch.masked_select(
                source,
                torch.tensor(
                    [
                        [True, False, False],
                        [True, False, True],
                        [False, True, True],
                    ]
                ),
            ),
            [0, 3, 5, 7, 8],
        )
