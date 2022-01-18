import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs, torch_eggs


class UnsqueezeTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.unsqueeze.html"
    TARGET = torch.unsqueeze

    def test_scalar(self) -> None:
        source = torch.tensor(3)

        torch_eggs.assert_view_tensor(
            torch.unsqueeze(source, 0),
            source,
            [3],
        )

    def test_unsqueeze(self) -> None:
        source = torch.tensor([1, 2, 3, 4])

        torch_eggs.assert_view_tensor(
            torch.unsqueeze(source, 0),
            source,
            [[1, 2, 3, 4]],
        )

        torch_eggs.assert_view_tensor(
            torch.unsqueeze(source, 1),
            source,
            [[1], [2], [3], [4]],
        )

        # negative dims

        torch_eggs.assert_view_tensor(
            torch.unsqueeze(source, -1),
            source,
            [[1], [2], [3], [4]],
        )

        torch_eggs.assert_view_tensor(
            torch.unsqueeze(source, -2),
            source,
            [[1, 2, 3, 4]],
        )

    def test_error(self) -> None:
        source = torch.tensor([1, 2, 3, 4])

        eggs.assert_raises(
            lambda: torch.unsqueeze(source, 2), IndexError, "Dimension out of range"
        )

        eggs.assert_raises(
            lambda: torch.unsqueeze(source, -3), IndexError, "Dimension out of range"
        )
