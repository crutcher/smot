import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import eggs


class NumelTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.numel.html"
    TARGET = torch.numel

    def test_numel(self) -> None:
        for s in [1, [1], [[1]]]:
            t = torch.tensor(s)
            eggs.assert_match(torch.numel(t), 1)
            eggs.assert_match(t.numel(), 1)

        for s in [[1, 2, 3, 4], [[1, 2], [3, 4]], [[1], [2], [3], [4]]]:
            t = torch.tensor(s)
            eggs.assert_match(torch.numel(t), 4)
            eggs.assert_match(t.numel(), 4)

        # Errors:
        # =======
        # Throws RuntimeError if type is not floating point.
        eggs.assert_raises(
            lambda: torch.numel([1, 2]),  # type: ignore
            TypeError,
        )
