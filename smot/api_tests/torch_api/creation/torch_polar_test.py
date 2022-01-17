import numpy as np
import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class PolarTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.polar.html"
    TARGET = torch.polar

    def test_polar(self) -> None:
        abs = torch.tensor([1, 2, 1], dtype=torch.float64)
        angle = torch.tensor(
            [np.pi / 2, 2 * np.pi, 5 * np.pi / 2],
            dtype=torch.float64,
        )

        torch_eggs.assert_tensor_close(
            torch.polar(abs, angle),
            torch.tensor(
                [0.0 + 1.0j, 2.0, 0.0 + 1.0j],
                dtype=torch.complex128,
            ),
        )
