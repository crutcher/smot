import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class TransposeTest(TorchApiTestCase):
    API_DOCS = "https://pytorch.org/docs/stable/generated/torch.transpose.html"
    TARGET = torch.transpose

    def test_transpose(self):
        source = torch.tensor(
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
        )
        torch_eggs.assert_view_tensor(
            torch.transpose(source, 0, 1),
            source,
            [
                [0, 3],
                [1, 4],
                [2, 5],
            ],
        )

    def test_error(self):
        source = torch.ones(2, 3)
        eggs.assert_raises(
            lambda: torch.transpose(source, 0, 3),
            IndexError,
            "Dimension out of range",
        )

    def test_transpose_3d(self):
        source = torch.tensor(
            [
                [
                    [[0, 1], [2, 3], [4, 5]],
                    [[6, 7], [8, 9], [10, 11]],
                ]
            ],
        )

        torch_eggs.assert_view_tensor(
            torch.transpose(source, 0, 2),
            source,
            [
                [
                    [[0, 1]],
                    [[6, 7]],
                ],
                [
                    [[2, 3]],
                    [[8, 9]],
                ],
                [
                    [[4, 5]],
                    [[10, 11]],
                ],
            ],
        )
