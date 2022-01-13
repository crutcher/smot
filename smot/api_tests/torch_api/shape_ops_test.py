import unittest

import torch

from smot.testlib import eggs


class ShapeOpsTest(unittest.TestCase):
    def test_numel(self):
        """torch.numel(input: Tensor)
        Also: `<tensor>.numel()`

        Returns the number of elements in the tensor.

        .. _Online Doc:
            https://pytorch.org/docs/stable/generated/torch.numel.html
        """
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
