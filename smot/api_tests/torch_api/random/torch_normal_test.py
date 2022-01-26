import unittest

import hamcrest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.multinomial",
    ref="https://pytorch.org/docs/stable/generated/torch.normal.html",
    note="When `std` is a cuda tensor, this function syncs with the CPU.",
)
class NormalTest(unittest.TestCase):
    def test_degenerate(self) -> None:
        torch_eggs.assert_tensor(
            torch.normal(torch.tensor([]), torch.tensor([])),
            [],
        )

        torch_eggs.assert_tensor(
            torch.normal(torch.tensor([[]]), torch.tensor([])),
            [[]],
        )

    def test_scalar(self) -> None:
        mean = 3.5
        std = 1.2
        k = 1000
        sample = torch.normal(mean, std, [k])
        eggs.assert_match(
            torch.mean(sample).item(),
            hamcrest.close_to(mean, k * 0.05),
        )
        eggs.assert_match(
            torch.std(sample).item(),
            hamcrest.close_to(std, k * 0.05),
        )

    def test_broadcast_shape(self) -> None:
        # the returned result is the broadcast shape of mean and stddev.

        torch_eggs.assert_tensor_structure(
            torch.normal(
                torch.tensor([1.0, 3.0]),
                torch.tensor([[[4.0], [3.0], [9.0]]]),
            ),
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
        )
