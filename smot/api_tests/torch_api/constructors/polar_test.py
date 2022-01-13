import unittest

import numpy as np
import torch

from smot.testlib import torch_eggs


class PolarTest(unittest.TestCase):
    # https://pytorch.org/docs/stable/generated/torch.polar.html

    def test_polar(self):
        abs = torch.tensor([1, 2, 1], dtype=torch.float64)
        angle = torch.tensor([np.pi / 2, 2 * np.pi, 5 * np.pi / 2], dtype=torch.float64)

        torch_eggs.assert_tensor_close(
            torch.polar(abs, angle),
            torch.tensor(
                [0.0 + 1.0j, 2.0, 0.0 + 1.0j],
                dtype=torch.complex128,
            ),
        )
