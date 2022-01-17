import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class PermuteTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.permute.html"
    TARGET = torch.permute

    def test_permute(self):
        source = torch.tensor(
            [
                [
                    [1, 2],
                    [3, 4],
                ],
                [
                    [5, 6],
                    [7, 8],
                ],
            ]
        )

        view = torch.permute(source, (2, 0, 1))

        torch_eggs.assert_views(source, view)

        torch_eggs.assert_tensor(
            view,
            [
                [
                    [1, 3],
                    [5, 7],
                ],
                [
                    [2, 4],
                    [6, 8],
                ],
            ],
        )
