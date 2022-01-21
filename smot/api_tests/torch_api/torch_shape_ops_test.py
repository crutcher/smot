import unittest

import torch

from smot.api_tests.doc_links import api_link
from smot.testlib import eggs


@api_link(
    target="torch.numel",
    ref="https://pytorch.org/docs/stable/generated/torch.numel.html",
)
class NumelTest(unittest.TestCase):
    def test_numel(self) -> None:
        for s in [1, [1], [[1]]]:
            t = torch.tensor(s)
            eggs.assert_match(torch.numel(t), 1)
            eggs.assert_match(t.numel(), 1)

        for s in [[1, 2, 3, 4], [[1, 2], [3, 4]], [[1], [2], [3], [4]]]:
            t = torch.tensor(s)
            eggs.assert_match(torch.numel(t), 4)
            eggs.assert_match(t.numel(), 4)

        # Errors:
        # =======
        # Throws RuntimeError if type is not floating point.
        eggs.assert_raises(
            lambda: torch.numel([1, 2]),  # type: ignore
            TypeError,
        )
