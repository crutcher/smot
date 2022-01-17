import torch

from smot.api_tests.torch_api.torch_api_testcase import TorchApiTestCase
from smot.testlib import torch_eggs


class FrombufferTest(TorchApiTestCase):
    API_DOC = "https://pytorch.org/docs/stable/generated/torch.frombuffer.html"
    TARGET = torch.frombuffer

    def test_frombuffer(self):
        source = bytearray([0, 1, 2, 3, 4])

        view = torch.frombuffer(
            source,
            count=3,
            offset=1,
            dtype=torch.int8,
        )

        torch_eggs.assert_tensor(
            view,
            torch.tensor([1, 2, 3], dtype=torch.int8),
        )

        # Mutations to one mutate the other.
        source[1] = 8

        torch_eggs.assert_tensor(
            view,
            torch.tensor([8, 2, 3], dtype=torch.int8),
        )

        # Anything that re-allocates the buffer is not safe,
        # it the torch pointer won't get updated to the new buffer.
        source.extend([20, 21, 22, 23] * 250)
        try:
            torch_eggs.assert_tensor(
                view,
                torch.tensor([2, 3, 4], dtype=torch.int8),
            )
        except AssertionError:
            # This is expected to fail.
            pass
