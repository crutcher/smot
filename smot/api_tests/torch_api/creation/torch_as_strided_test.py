import unittest

import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import eggs, torch_eggs


@api_link(
    target="torch.as_strided",
    ref="https://pytorch.org/docs/stable/generated/torch.as_strided.html",
)
class AsStridedTest(unittest.TestCase):
    def test_as_strided(self) -> None:
        t = torch.rand(9)
        v = t.data

        x = torch.as_strided(t, (2, 2), (1, 2))

        eggs.assert_match(
            x.data_ptr(),
            t.data_ptr(),
        )

        torch_eggs.assert_tensor_equals(
            x,
            [
                [v[1 * 0 + 2 * 0], v[1 * 0 + 2 * 1]],
                [v[1 * 1 + 2 * 0], v[1 * 1 + 2 * 1]],
            ],
        )

    def test_storage_offset(self) -> None:
        t = torch.rand(9, dtype=torch.float32)
        v = t.data
        offset = 2

        x = torch.as_strided(t, (2, 2), (1, 2), 2)

        eggs.assert_match(
            x.data_ptr(),
            t.data_ptr() + 4 * offset,
        )

        torch_eggs.assert_tensor_equals(
            x,
            [
                [v[2 + 1 * 0 + 2 * 0], v[2 + 1 * 0 + 2 * 1]],
                [v[2 + 1 * 1 + 2 * 0], v[2 + 1 * 1 + 2 * 1]],
            ],
        )
