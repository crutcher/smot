import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.dstack",
    ref="https://pytorch.org/docs/stable/generated/torch.dstack.html",
)
class DstackTest(unittest.TestCase):
    def test_dstack(self) -> None:
        torch_eggs.assert_tensor_equals(
            torch.dstack(
                (
                    torch.tensor([1, 2, 3]),
                    torch.tensor([4, 5, 6]),
                )
            ),
            [
                [
                    [1, 4],
                    [2, 5],
                    [3, 6],
                ]
            ],
        )

        torch_eggs.assert_tensor_equals(
            torch.dstack(
                (
                    torch.tensor([[1], [2], [3]]),
                    torch.tensor([[4], [5], [6]]),
                )
            ),
            [
                [[1, 4]],
                [[2, 5]],
                [[3, 6]],
            ],
        )

        torch_eggs.assert_tensor_equals(
            torch.dstack(
                (
                    torch.tensor([[[1, 2]], [[3, 4]]]),
                    torch.tensor([[5], [6]]),
                )
            ),
            [
                [[1, 2, 5]],
                [[3, 4, 6]],
            ],
        )

    def test_out(self) -> None:
        target = torch.arange(6)
        orig_data_ptr = target.data_ptr()

        torch.dstack(
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
            ),
            out=target,
        )

        torch_eggs.assert_tensor_equals(
            target,
            [
                [
                    [1, 4],
                    [2, 5],
                    [3, 6],
                ]
            ],
        )

        eggs.assert_match(target.data_ptr(), orig_data_ptr)
