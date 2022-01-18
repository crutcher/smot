import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class VsplitTest(TorchApiTestCase):
    API_DOCS = "https://pytorch.org/docs/stable/generated/torch.vsplit.html"
    TARGET = torch.vsplit

    def test_vsplit(self) -> None:
        source = torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ]
        )

        torch_eggs.assert_view_tensor_seq(
            torch.vsplit(source, 2),
            source,
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ],
            [
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.vsplit(source, 4),
            source,
            [
                [0, 1, 2, 3],
            ],
            [
                [4, 5, 6, 7],
            ],
            [
                [8, 9, 10, 11],
            ],
            [
                [12, 13, 14, 15],
            ],
        )

        torch_eggs.assert_view_tensor_seq(
            torch.vsplit(source, (1, 3)),
            source,
            [
                [0, 1, 2, 3],
            ],
            [
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            [
                [12, 13, 14, 15],
            ],
        )

    def test_error(self) -> None:
        source = torch.tensor(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ]
        )

        eggs.assert_raises(
            lambda: torch.vsplit(source, 3),
            RuntimeError,
            "the size of the dimension 4 is not divisible by",
        )
