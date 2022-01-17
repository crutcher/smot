import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class HeavisideTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.heaviside.html"
    TARGET = torch.heaviside

    def test_heaviside_scalar_value(self):
        input = torch.tensor([-1.5, 0, 2.0])
        values = torch.tensor(0.5)
        torch_eggs.assert_tensor(
            torch.heaviside(input, values),
            torch.tensor([0.0, 0.5, 1.0]),
        )

    def test_heaviside_single_value(self):
        input = torch.tensor([-1.5, 0, 2.0])
        values = torch.tensor([0.5])
        torch_eggs.assert_tensor(
            torch.heaviside(input, values),
            torch.tensor([0.0, 0.5, 1.0]),
        )

    def test_heaviside_multi_value(self):
        input = torch.tensor([-1.5, 0, 0])

        eggs.assert_raises(
            lambda: torch.heaviside(input, torch.tensor([0.5, 2])),
            RuntimeError,
            r"must match the size of tensor b \(2\) at non-singleton dimension 0",
        )

        torch_eggs.assert_tensor(
            torch.heaviside(input, torch.tensor([5.0, 6.0, 7.0])),
            torch.tensor([0.0, 6.0, 7.0]),
        )
