import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class MaskedSelectTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.masked_select.html"
    TARGET = torch.masked_select

    def test_select(self) -> None:
        source = torch.arange(9).reshape(3, 3)

        torch_eggs.assert_tensor(
            source,
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
        )

        torch_eggs.assert_tensor(
            torch.masked_select(
                source,
                torch.tensor(
                    [
                        [True, False, False],
                        [True, False, True],
                        [False, True, True],
                    ]
                ),
            ),
            [0, 3, 5, 7, 8],
        )
