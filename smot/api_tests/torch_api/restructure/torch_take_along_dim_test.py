import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class TakeAlongDimTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.take_along_dim.html"
    TARGET = torch.take_along_dim

    def test_take(self) -> None:
        src = torch.tensor([[10, 30, 20], [60, 40, 50]])

        max_idx = torch.argmax(src)
        torch_eggs.assert_tensor(
            max_idx,
            # no dim, acts like take
            3,
        )
        torch_eggs.assert_tensor(
            torch.take_along_dim(src, max_idx),
            [60],
        )

        sorted_idx = torch.argsort(src, dim=1)
        torch_eggs.assert_tensor(
            sorted_idx,
            [[0, 2, 1], [1, 2, 0]],
        )
        torch_eggs.assert_tensor(
            torch.take_along_dim(src, sorted_idx, dim=1),
            [[10, 20, 30], [40, 50, 60]],
        )

    def test_error(self) -> None:
        src = torch.tensor([[10, 30, 20], [60, 40, 50]])

        eggs.assert_raises(
            lambda: torch.take_along_dim(src, torch.tensor([3]), dim=2),
            RuntimeError,
            "input and indices should have the same number of dimensions",
        )
