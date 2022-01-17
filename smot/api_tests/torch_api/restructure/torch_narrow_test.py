import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class NarrowTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.narrow.html"
    TARGET = torch.narrow

    def test_narrow(self) -> None:
        source = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )

        view = torch.narrow(source, 0, 1, 2)

        torch_eggs.assert_views(source, view)

        torch_eggs.assert_tensor(
            view,
            [
                [4, 5, 6],
                [7, 8, 9],
            ],
        )

        view = torch.narrow(source, 1, 0, 1)

        torch_eggs.assert_views(source, view)

        torch_eggs.assert_tensor(
            view,
            [
                [1],
                [4],
                [7],
            ],
        )
