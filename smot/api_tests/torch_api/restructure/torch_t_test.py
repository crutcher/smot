import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class TTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.t.html"
    TARGET = torch.t

    def test_t_scalar(self) -> None:
        # <= 2 dimensions ...
        source = torch.tensor(3)
        torch_eggs.assert_view_tensor(
            torch.t(source),
            source,
            3,
        )

    def test_t_1d(self) -> None:
        # <= 2 dimensions ...
        source = torch.tensor([3, 2])
        torch_eggs.assert_view_tensor(
            torch.t(source),
            source,
            [3, 2],
        )

    def test_t(self) -> None:
        source = torch.arange(6).reshape(2, 3)
        torch_eggs.assert_tensor(
            source,
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
        )

        torch_eggs.assert_view_tensor(
            torch.t(source),
            source,
            [
                [0, 3],
                [1, 4],
                [2, 5],
            ],
        )

    def test_error(self) -> None:
        source = torch.arange(6).reshape(2, 1, 3)
        eggs.assert_raises(
            lambda: torch.t(source),
            RuntimeError,
            "expects a tensor with <= 2 dimensions",
        )
