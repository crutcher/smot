import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class TakeTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.take.html"
    TARGET = torch.take

    def test_take(self) -> None:
        src = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )

        torch_eggs.assert_tensor(
            torch.take(src, torch.tensor([0, 2, 3])),
            [1, 3, 4],
        )

        torch_eggs.assert_tensor(
            torch.take(src, torch.tensor([[0, 2], [3, 5]])),
            [[1, 3], [4, 6]],
        )

        view = src.t()
        torch_eggs.assert_view_tensor(
            view,
            src,
            [[1, 4], [2, 5], [3, 6]],
        )

        # indexing is done 1-d in the view's indexing.

        torch_eggs.assert_tensor(
            torch.take(view, torch.tensor([0, 2, 3])),
            [1, 2, 5],
        )

        torch_eggs.assert_tensor(
            torch.take(view, torch.tensor([[0, 2], [3, 5]])),
            [[1, 2], [5, 6]],
        )