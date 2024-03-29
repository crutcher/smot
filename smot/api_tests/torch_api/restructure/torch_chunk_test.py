import unittest

import hamcrest
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.chunk",
    ref="https://pytorch.org/docs/stable/generated/torch.chunk.html",
)
class ChunkTest(unittest.TestCase):
    def test_chunk(self) -> None:
        source = torch.tensor([1, 2, 3, 4, 5, 6])
        chunks = torch.chunk(source, 2)

        torch_eggs.assert_tensor_views(source, *chunks)

        eggs.assert_match(
            chunks,
            hamcrest.contains_exactly(
                torch_eggs.matches_tensor([1, 2, 3]),
                torch_eggs.matches_tensor([4, 5, 6]),
            ),
        )

        # These chunks are bi-dir views.
        source[0] = 88
        source[3] = 99

        chunks[0][1] = 66
        chunks[1][1] = 77

        eggs.assert_match(
            chunks,
            hamcrest.contains_exactly(
                torch_eggs.matches_tensor([88, 66, 3]),
                torch_eggs.matches_tensor([99, 77, 6]),
            ),
        )

        torch_eggs.assert_tensor_equals(
            source,
            [88, 66, 3, 99, 77, 6],
        )

    def test_chunk_dims(self) -> None:
        source = torch.tensor([[1, 2, 3], [4, 5, 6]])

        eggs.assert_match(
            torch.chunk(source, 3, dim=1),
            hamcrest.contains_exactly(
                torch_eggs.matches_tensor([[1], [4]]),
                torch_eggs.matches_tensor([[2], [5]]),
                torch_eggs.matches_tensor([[3], [6]]),
            ),
        )

        eggs.assert_match(
            torch.chunk(source, 2, dim=1),
            hamcrest.contains_exactly(
                torch_eggs.matches_tensor([[1, 2], [4, 5]]),
                torch_eggs.matches_tensor([[3], [6]]),
            ),
        )

    def test_trailing_chunk(self) -> None:
        source = torch.tensor([1, 2, 3, 4, 5, 6])

        # returning less than requested chunks,
        # because 6 // 4 = 1
        # 1 * 4 < 6
        # (1 + 1) * 4 = 8 > 6

        # all chunks must be the same size, except for the last one,
        # which may be smaller.
        # all chunks must be an integer size.
        # "empty" chunks are omitted.

        eggs.assert_match(
            torch.chunk(
                torch.tensor([1, 2, 3, 4, 5, 6]),
                4,
            ),
            hamcrest.contains_exactly(
                torch_eggs.matches_tensor([1, 2]),
                torch_eggs.matches_tensor([3, 4]),
                torch_eggs.matches_tensor([5, 6]),
            ),
        )

        eggs.assert_match(
            torch.chunk(
                torch.tensor([1, 2, 3, 4, 5]),
                4,
            ),
            hamcrest.contains_exactly(
                torch_eggs.matches_tensor([1, 2]),
                torch_eggs.matches_tensor([3, 4]),
                torch_eggs.matches_tensor([5]),
            ),
        )

        eggs.assert_match(
            torch.chunk(
                torch.tensor([1, 2, 3, 4, 5, 6, 7]),
                4,
            ),
            hamcrest.contains_exactly(
                torch_eggs.matches_tensor([1, 2]),
                torch_eggs.matches_tensor([3, 4]),
                torch_eggs.matches_tensor([5, 6]),
                torch_eggs.matches_tensor([7]),
            ),
        )
