import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class StackTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.stack.html"
    TARGET = torch.stack

    def test_stack(self) -> None:
        a = torch.arange(9, dtype=torch.int64).reshape(3, 3)
        a += 10

        b = torch.arange(9, dtype=torch.int64).reshape(3, 3)
        b += 20

        torch_eggs.assert_tensor(
            torch.stack((a, b)),
            [
                [
                    [10, 11, 12],
                    [13, 14, 15],
                    [16, 17, 18],
                ],
                [
                    [20, 21, 22],
                    [23, 24, 25],
                    [26, 27, 28],
                ],
            ],
        )

        torch_eggs.assert_tensor(
            torch.stack((a, b), dim=1),
            [
                [
                    [10, 11, 12],
                    [20, 21, 22],
                ],
                [
                    [13, 14, 15],
                    [23, 24, 25],
                ],
                [
                    [16, 17, 18],
                    [26, 27, 28],
                ],
            ],
        )

        torch_eggs.assert_tensor(
            torch.stack((a, b), dim=2),
            [
                [[10, 20], [11, 21], [12, 22]],
                [[13, 23], [14, 24], [15, 25]],
                [[16, 26], [17, 27], [18, 28]],
            ],
        )

    def test_errors(self) -> None:
        a = torch.ones(3, 2)
        b = torch.ones(2, 2)

        eggs.assert_raises(
            lambda: torch.stack((a, a), 3),
            IndexError,
            "Dimension out of range",
        )

        eggs.assert_raises(
            lambda: torch.stack((a, b)),
            RuntimeError,
            "expects each tensor to be equal size",
        )
