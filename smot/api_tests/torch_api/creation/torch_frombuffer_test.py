import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.frombuffer",
    ref="https://pytorch.org/docs/stable/generated/torch.frombuffer.html",
)
class FrombufferTest(unittest.TestCase):
    def test_frombuffer(self) -> None:
        source = bytearray([0, 1, 2, 3, 4])

        view = torch.frombuffer(
            source,
            count=3,
            offset=1,
            dtype=torch.int8,
        )

        torch_eggs.assert_tensor(
            view,
            torch.tensor([1, 2, 3], dtype=torch.int8),
        )

        # Mutations to one mutate the other.
        source[1] = 8

        torch_eggs.assert_tensor(
            view,
            torch.tensor([8, 2, 3], dtype=torch.int8),
        )

        # Anything that re-allocates the buffer is not safe,
        # it the torch pointer won't get updated to the new buffer.
        source.extend([20, 21, 22, 23] * 250)
        try:
            torch_eggs.assert_tensor(
                view,
                torch.tensor([2, 3, 4], dtype=torch.int8),
            )
        except AssertionError:
            # This is expected to fail.
            pass
