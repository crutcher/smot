import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class ScatterAddTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.scatter_add.html"
    TARGET = torch.scatter_add

    # Note: the backward pass is implemented only for src.shape == index.shape

    def test_add(self) -> None:
        source = torch.arange(9, dtype=torch.int64).reshape(3, 3)

        torch_eggs.assert_tensor(
            torch.scatter_add(
                source,
                1,
                torch.tensor(
                    [
                        [2],
                        [1],
                        [0],
                    ],
                ),
                source,
            ),
            [
                [0, 1, 2],
                [3, 7, 5],
                [12, 7, 8],
            ],
        )
