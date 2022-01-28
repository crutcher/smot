import unittest

import numpy as np
import torch

from smot.doc_link.link_annotations import api_link
from smot.testlib import torch_eggs


@api_link(
    target="torch.polar",
    ref="https://pytorch.org/docs/stable/generated/torch.polar.html",
)
class PolarTest(unittest.TestCase):
    def test_polar(self) -> None:
        abs = torch.tensor([1, 2, 1], dtype=torch.float64)
        angle = torch.tensor(
            [np.pi / 2, 2 * np.pi, 5 * np.pi / 2],
            dtype=torch.float64,
        )

        torch_eggs.assert_tensor_equals(
            torch.polar(abs, angle),
            torch.tensor(
                [0.0 + 1.0j, 2.0, 0.0 + 1.0j],
                dtype=torch.complex128,
            ),
            close=True,
        )
